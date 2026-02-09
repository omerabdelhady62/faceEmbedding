import React, { useRef, useState } from 'react'
import {
  ActivityIndicator,
  Button,
  StyleSheet,
  Text,
  useColorScheme,
  View,
  StatusBar,
  Image,
} from 'react-native'
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context'
import FaceDetection from '@react-native-ml-kit/face-detection'
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite'
import RNFS from 'react-native-fs'
import ImageEditor from '@react-native-community/image-editor'

// Pixel decoding for a static image URI
// Install these (pure JS) decoders:
//   yarn add jpeg-js pngjs
// Note: WebP is NOT supported by these decoders. Use PNG/JPG assets for this screen.
import jpeg from 'jpeg-js'
import { PNG } from 'pngjs/browser'

import { Buffer } from 'buffer'
  ; (globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

function nowMs(): number {
  const p = (globalThis as any)?.performance
  return typeof p?.now === 'function' ? p.now() : Date.now()
}

/**
 * Handles both remote URLs and local require() assets.
 * Returns a local file:// URI.
 */
async function resolveImageToLocalUri(source: any, tag: string): Promise<string> {
  let uri: string
  if (typeof source === 'number') {
    const resolved = Image.resolveAssetSource(source)
    uri = resolved.uri
  } else if (typeof source === 'string') {
    uri = source
  } else {
    throw new Error(`Unsupported image source type for ${tag}`)
  }

  const isPng = uri.toLowerCase().includes('.png')
  const isJpg = uri.toLowerCase().includes('.jpg') || uri.toLowerCase().includes('.jpeg')
  if (!isPng && !isJpg) {
    throw new Error(
      `Unsupported image extension for ${tag}. Use .png/.jpg (WebP not supported by the JS decoders).`,
    )
  }

  const extension = isPng ? 'png' : 'jpg'
  const destPath = `${RNFS.TemporaryDirectoryPath}/img_${tag}_${Date.now()}.${extension}`

  // downloadFile also works for packager-served assets
  const res = await RNFS.downloadFile({ fromUrl: uri, toFile: destPath }).promise
  if (res.statusCode && res.statusCode >= 400) {
    throw new Error(`Failed to resolve image ${tag} (status ${res.statusCode})`)
  }
  return `file://${destPath}`
}

type FaceFrame = { x: number; y: number; width: number; height: number }

async function detectFirstFaceFrame(localFileUri: string): Promise<FaceFrame> {
  const faces = await FaceDetection.detect(localFileUri, {
    landmarkMode: 'all',
    trackingEnabled: false,
    performanceMode: 'fast',
  })
  if (!faces || faces.length === 0) throw new Error(`No face detected for: ${localFileUri}`)
  const frame = faces[0]?.frame
  if (!frame) throw new Error(`Face bbox(frame) missing for: ${localFileUri}`)
  return frame
}

/**
 * Replicates the Android repo behavior: make the detected face bbox a square.
 * (Repo: Box.toSquareShape() + limitSquare() before cropping.)
 */
function makeSquareFrame(frame: FaceFrame): FaceFrame {
  const cx = frame.x + frame.width / 2
  const cy = frame.y + frame.height / 2
  const size = Math.max(frame.width, frame.height)
  return { x: cx - size / 2, y: cy - size / 2, width: size, height: size }
}

/**
 * Crop square face and resize to 256x256 (repo input size).
 */
async function cropAndResize256(localFileUri: string, square: FaceFrame): Promise<string> {
  const offset = { x: Math.max(0, Math.round(square.x)), y: Math.max(0, Math.round(square.y)) }
  const size = { width: Math.max(1, Math.round(square.width)), height: Math.max(1, Math.round(square.height)) }

  // ✅ FIX: Extract .uri from the result object
  const result = await ImageEditor.cropImage(localFileUri, {
    offset,
    size,
    displaySize: { width: 256, height: 256 },
    resizeMode: 'stretch',
  })

  const croppedUri = typeof result === 'string' ? result : result.uri;

  if (!croppedUri) throw new Error('ImageEditor.cropImage returned empty uri');

  // Ensure the file:// prefix is present for RNFS
  return croppedUri.startsWith('file://') ? croppedUri : `file://${croppedUri}`;
}

type DecodedRGBA = { width: number; height: number; data: Uint8Array }

async function decodeImageFileToRgba(fileUri: any): Promise<DecodedRGBA> {
  // ✅ FIX: Ensure we are working with a string
  const uriString = typeof fileUri === 'string' ? fileUri : fileUri?.uri;

  if (!uriString || typeof uriString !== 'string' || !uriString.startsWith('file://')) {
    throw new Error(`Expected file:// uri string, got: ${JSON.stringify(fileUri)}`);
  }

  const path = uriString.replace('file://', '')
  const base64 = await RNFS.readFile(path, 'base64')
  const buf = Buffer.from(base64, 'base64')
  const lower = path.toLowerCase()

  // Some RN image pipelines return a temp file without an extension.
  // Try to detect by magic header first; fall back to extension.
  const isJpgByMagic = buf.length >= 2 && buf[0] === 0xff && buf[1] === 0xd8
  const isPngByMagic =
    buf.length >= 8 &&
    buf[0] === 0x89 &&
    buf[1] === 0x50 &&
    buf[2] === 0x4e &&
    buf[3] === 0x47 &&
    buf[4] === 0x0d &&
    buf[5] === 0x0a &&
    buf[6] === 0x1a &&
    buf[7] === 0x0a

  const isJpgByExt = lower.endsWith('.jpg') || lower.endsWith('.jpeg')
  const isPngByExt = lower.endsWith('.png')

  if (isJpgByMagic || isJpgByExt) {
    const decoded = jpeg.decode(buf, { useTArray: true }) as unknown as {
      width: number
      height: number
      data: Uint8Array
    }
    return { width: decoded.width, height: decoded.height, data: decoded.data }
  }

  if (isPngByMagic || isPngByExt) {
    const decoded = PNG.sync.read(buf)
    // decoded.data is Buffer/Uint8Array RGBA
    return { width: decoded.width, height: decoded.height, data: decoded.data as unknown as Uint8Array }
  }

  throw new Error(`Unsupported image type: ${path}`)
}

/**
 * Repo equivalent:
 * - resize to 256x256
 * - normalize RGB by /255
 * - NHWC float32
 */
function rgba256ToRgbFloatNHWC01(rgba: Uint8Array): Float32Array {
  // rgba is length 256*256*4
  if (rgba.length !== 256 * 256 * 4) {
    throw new Error(`Expected RGBA length ${256 * 256 * 4}, got ${rgba.length}`)
  }
  const out = new Float32Array(256 * 256 * 3)
  let j = 0
  for (let i = 0; i < rgba.length; i += 4) {
    // RGBA -> RGB
    out[j++] = rgba[i] / 255.0
    out[j++] = rgba[i + 1] / 255.0
    out[j++] = rgba[i + 2] / 255.0
  }
  return out
}

/**
 * Repo score combine:
 * score = Σ_i ( abs(clss_pred[i]) * leaf_node_mask[i] )
 * live if score < 0.2
 */
function computeAntiSpoofScore(clssPred: Float32Array, leafMask: Float32Array): number {
  if (clssPred.length !== leafMask.length) {
    throw new Error(`Output length mismatch: ${clssPred.length} vs ${leafMask.length}`)
  }
  let s = 0
  for (let i = 0; i < clssPred.length; i++) {
    s += Math.abs(clssPred[i]) * leafMask[i]
  }
  return s
}

function App() {
  const isDarkMode = useColorScheme() === 'dark'
  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <AppContent />
    </SafeAreaProvider>
  )
}

function AppContent() {
  const insets = useSafeAreaInsets()
  const [loading, setLoading] = useState(false)
  const [output, setOutput] = useState('')
  const modelRef = useRef<TensorflowModel | null>(null)

  const getAntiSpoofModel = async (): Promise<TensorflowModel> => {
    if (modelRef.current) return modelRef.current
    // Put FaceAntiSpoofing.tflite under ./model/
    modelRef.current = await loadTensorflowModel(require('./model/FaceAntiSpoofing.tflite'))
    return modelRef.current
  }

  const handleRunSingleImage = async () => {
    // Use PNG/JPG here.
    const source = require('./assets/ana.jpeg')
    const SPOOF_THRESHOLD = 0.2

    try {
      setLoading(true)
      setOutput('')

      const img = await resolveImageToLocalUri(source, 'img')

      const t0 = nowMs()
      const frame = await detectFirstFaceFrame(img)
      const t1 = nowMs()

      const square = makeSquareFrame(frame)
      const face256 = await cropAndResize256(img, square)

      // Decode 256x256 RGBA
      const decoded = await decodeImageFileToRgba(face256)
      if (decoded.width !== 256 || decoded.height !== 256) {
        throw new Error(`Expected 256x256 after crop+resize, got ${decoded.width}x${decoded.height}`)
      }

      const input = rgba256ToRgbFloatNHWC01(decoded.data)

      const model = await getAntiSpoofModel()
      const t2 = nowMs()

      // IMPORTANT:
      // This model has TWO outputs. react-native-fast-tflite returns outputs in the model's output order.
      // In the original Android code, they map outputs by tensor names "Identity" and "Identity_1".
      const outputs = await model.run([input])

      // Most TFLite wrappers return raw typed arrays OR ArrayBuffers.
      // We normalize to Float32Array.
      const o0 = outputs?.[0]
      const o1 = outputs?.[1]
      if (!o0 || !o1) {
        throw new Error(
          `Model did not return 2 outputs. Got ${Array.isArray(outputs) ? outputs.length : 'non-array'}.`,
        )
      }

      const clssPred = o0 instanceof Float32Array ? o0 : new Float32Array(o0)
      const leafMask = o1 instanceof Float32Array ? o1 : new Float32Array(o1)

      const score = computeAntiSpoofScore(clssPred, leafMask)
      const t3 = nowMs()

      const verdict = score < SPOOF_THRESHOLD ? '✅ LIVE' : '❌ SPOOF'

      setOutput(
        [
          `Face detected.`,
          `Anti-spoof score: ${score.toFixed(6)} (threshold ${SPOOF_THRESHOLD})`,
          `Decision: ${verdict}`,
          '',
          `Detection time: ${(t1 - t0).toFixed(2)} ms`,
          `Inference time: ${(t3 - t2).toFixed(2)} ms`,
          '',
          `Debug: face256 uri: ${face256}`,
        ].join('\n'),
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <View style={styles.buttonWrap}>
        <Button title="Run Anti-Spoof on 1 image" onPress={handleRunSingleImage} disabled={loading} />
      </View>
      {loading && <ActivityIndicator size="small" style={styles.loader} />}
      {!!output && <Text style={styles.result}>{output}</Text>}
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, backgroundColor: '#fff' },
  buttonWrap: { marginTop: 8, marginBottom: 16 },
  loader: { marginVertical: 8 },
  result: { fontSize: 12, fontWeight: '600', color: '#333', lineHeight: 18 },
})

export default App
