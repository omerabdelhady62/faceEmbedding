import React, { useRef, useState } from 'react'
import {
  ActivityIndicator,
  Button,
  StyleSheet,
  Text,
  useColorScheme,
  View,
  StatusBar,
  Image, // Added for resolveAssetSource
} from 'react-native'
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context'
import FaceDetection from '@react-native-ml-kit/face-detection'
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite'
import RNFS from 'react-native-fs'
import ImageEditor from '@react-native-community/image-editor'
import { Buffer } from 'buffer'

  // ✅ IMPORTANT: polyfill Buffer for the environment
  ; (globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

import {
  getMobileFaceNetEmbeddingFromFrame,
  cropFace112ForDebug,
  type FaceFrame,
} from './faceEmbedding.ts'

function nowMs(): number {
  const p = (globalThis as any)?.performance
  return typeof p?.now === 'function' ? p.now() : Date.now()
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`Embedding length mismatch: ${a.length} vs ${b.length}`)
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i]
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

/**
 * ✅ FIX: Handles both remote URLs and local require() assets
 */
async function resolveImageToLocalUri(source: any, tag: string): Promise<string> {
  let uri: string;

  if (typeof source === 'number') {
    // It's a local require() asset ID
    const resolved = Image.resolveAssetSource(source);
    uri = resolved.uri;
  } else if (typeof source === 'string') {
    // It's a remote URL
    uri = source;
  } else {
    throw new Error(`Unsupported image source type for ${tag}`);
  }

  const extension = uri.includes('.png') ? 'png' : 'jpg';
  const destPath = `${RNFS.TemporaryDirectoryPath}/img_${tag}_${Date.now()}.${extension}`;

  // We use downloadFile even for local assets because in Dev mode 
  // they are served over http from the Metro server.
  const res = await RNFS.downloadFile({
    fromUrl: uri,
    toFile: destPath
  }).promise;

  if (res.statusCode && res.statusCode >= 400) {
    throw new Error(`Failed to resolve image ${tag} (status ${res.statusCode})`);
  }

  return `file://${destPath}`;
}

async function saveFileUriToDocuments(fileUri: string, filename: string): Promise<string> {
  if (!fileUri.startsWith('file://')) {
    throw new Error(`Expected file:// URI to save, got: ${fileUri}`)
  }
  const srcPath = fileUri.replace('file://', '')
  const destPath = `${RNFS.DocumentDirectoryPath}/${filename}`
  await RNFS.copyFile(srcPath, destPath)
  return `file://${destPath}`
}

async function detectFirstFaceFrame(localFileUri: string): Promise<FaceFrame> {
  const faces = await FaceDetection.detect(localFileUri, { landmarkMode: 'all', trackingEnabled: false, performanceMode: 'fast' })
  if (!faces || faces.length === 0) throw new Error(`No face detected for: ${localFileUri}`)
  const frame = faces[0]?.frame
  if (!frame) throw new Error(`Face bbox(frame) missing for: ${localFileUri}`)
  return frame
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

  const getModel = async (): Promise<TensorflowModel> => {
    if (modelRef.current) return modelRef.current
    modelRef.current = await loadTensorflowModel(
      require('./model/mobilefacenet.tflite'),
    )
    return modelRef.current
  }

  const handleCompare = async () => {
    // Can be require() or 'https://...'
    const source1 = require('./assets/omar.png')
    // const source2 = require('./assets/omar3.jpeg')
    const source2 = require('./assets/omar2.webp')

    const THRESHOLD = 0.60

    try {
      setLoading(true)
      setOutput('')

      // ✅ Correctly resolve images to local file paths
      const img1 = await resolveImageToLocalUri(source1, 'img1')
      const img2 = await resolveImageToLocalUri(source2, 'img2')

      const model = await getModel()

      const tDet0 = nowMs()
      const [frame1, frame2] = await Promise.all([
        detectFirstFaceFrame(img1),
        detectFirstFaceFrame(img2),
      ])
      const tDet1 = nowMs()
      const detectMs = tDet1 - tDet0

      let savedCrop1: string | null = null
      let savedCrop2: string | null = null

      try {
        const crop1 = await cropFace112ForDebug(img1, frame1);
        const crop2 = await cropFace112ForDebug(img2, frame2);

        // Add this safety check
        if (!crop1 || !crop2) throw new Error("Cropping returned undefined");

        savedCrop1 = await saveFileUriToDocuments(crop1, `face_crop_1_${Date.now()}.jpg`);
        savedCrop2 = await saveFileUriToDocuments(crop2, `face_crop_2_${Date.now()}.jpg`);
      } catch (e) {
        console.warn('Saving cropped faces failed:', e);
      }

      const tEmb0 = nowMs()
      const [emb1, emb2] = await Promise.all([
        getMobileFaceNetEmbeddingFromFrame(img1, frame1, model),
        getMobileFaceNetEmbeddingFromFrame(img2, frame2, model),
      ])
      const tEmb1 = nowMs()
      const embedMs = tEmb1 - tEmb0

      const sim = cosineSimilarity(emb1, emb2)
      const verdict = sim >= THRESHOLD ? '✅ MATCH' : '❌ NO MATCH'

      setOutput(
        [
          `Similarity (cosine): ${sim.toFixed(6)}`,
          `Decision: ${verdict}`,
          '',
          `Detection: ${detectMs.toFixed(2)} ms`,
          `Embedding: ${embedMs.toFixed(2)} ms`,
          '',
          `Saved crop 1: ${savedCrop1 ?? 'N/A'}`,
          `Saved crop 2: ${savedCrop2 ?? 'N/A'}`,
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
        <Button title="Compare 2 images (timed)" onPress={handleCompare} disabled={loading} />
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