import FaceDetection from '@react-native-ml-kit/face-detection'
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite'
import RNFS from 'react-native-fs'
import ImageEditor from '@react-native-community/image-editor'

// JS decoders (same as the attached live-detection demo)
import jpeg from 'jpeg-js'
import { PNG } from 'pngjs/browser'

import { Buffer } from 'buffer'
;(globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

export type AnyMlKitFrame =
  | { x: number; y: number; width: number; height: number }
  | { left: number; top: number; width: number; height: number }

export type Xywh = { x: number; y: number; width: number; height: number }

export function normalizeFrameToXywh(frame: AnyMlKitFrame): Xywh {
  const anyF: any = frame
  const x = typeof anyF.x === 'number' ? anyF.x : anyF.left
  const y = typeof anyF.y === 'number' ? anyF.y : anyF.top
  if (![x, y, anyF.width, anyF.height].every((v) => typeof v === 'number')) {
    throw new Error(`Invalid face frame shape: ${JSON.stringify(frame)}`)
  }
  return { x, y, width: anyF.width, height: anyF.height }
}

/**
 * Replicates the Android repo behavior for anti-spoof:
 * - make face bbox a square (center-crop to square)
 * - crop and resize to 256x256 (stretch)
 * - normalize RGB to [0..1]
 * - NHWC float32
 */
function makeSquareFrame(frame: Xywh): Xywh {
  const cx = frame.x + frame.width / 2
  const cy = frame.y + frame.height / 2
  const size = Math.max(frame.width, frame.height)
  return { x: cx - size / 2, y: cy - size / 2, width: size, height: size }
}

async function cropAndResize256(localFileUri: string, square: Xywh): Promise<string> {
  if (!localFileUri.startsWith('file://')) {
    throw new Error(`antiSpoof expects file:// URI, got: ${localFileUri}`)
  }

  const offset = { x: Math.max(0, Math.round(square.x)), y: Math.max(0, Math.round(square.y)) }
  const size = {
    width: Math.max(1, Math.round(square.width)),
    height: Math.max(1, Math.round(square.height)),
  }

  const result = await ImageEditor.cropImage(localFileUri, {
    offset,
    size,
    displaySize: { width: 256, height: 256 },
    resizeMode: 'stretch',
  })

  const croppedUri = typeof result === 'string' ? result : result.uri
  if (!croppedUri) throw new Error('ImageEditor.cropImage returned empty uri')

  return croppedUri.startsWith('file://') ? croppedUri : `file://${croppedUri}`
}

type DecodedRGBA = { width: number; height: number; data: Uint8Array }

async function decodeImageFileToRgba(fileUri: string): Promise<DecodedRGBA> {
  if (!fileUri.startsWith('file://')) {
    throw new Error(`Expected file:// uri string, got: ${fileUri}`)
  }
  const path = fileUri.replace('file://', '')
  const base64 = await RNFS.readFile(path, 'base64')
  const buf = Buffer.from(base64, 'base64')
  const lower = path.toLowerCase()

  // Detect by magic header first; fall back to extension.
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
    return { width: decoded.width, height: decoded.height, data: decoded.data as unknown as Uint8Array }
  }

  throw new Error(`Unsupported image type for antiSpoof decode: ${path}`)
}

function rgba256ToRgbFloatNHWC01(rgba: Uint8Array): Float32Array {
  if (rgba.length !== 256 * 256 * 4) {
    throw new Error(`Expected RGBA length ${256 * 256 * 4}, got ${rgba.length}`)
  }
  const out = new Float32Array(256 * 256 * 3)
  let j = 0
  for (let i = 0; i < rgba.length; i += 4) {
    out[j++] = rgba[i] / 255.0
    out[j++] = rgba[i + 1] / 255.0
    out[j++] = rgba[i + 2] / 255.0
  }
  return out
}

function computeAntiSpoofScore(clssPred: Float32Array, leafMask: Float32Array): number {
  if (clssPred.length !== leafMask.length) {
    throw new Error(`Output length mismatch: ${clssPred.length} vs ${leafMask.length}`)
  }
  let s = 0
  for (let i = 0; i < clssPred.length; i++) s += Math.abs(clssPred[i]) * leafMask[i]
  return s
}

export type AntiSpoofResult = {
  score: number
  threshold: number
  isLive: boolean
  debug?: {
    face256Uri: string
  }
}

/**
 * Run anti-spoof check.
 *
 * Inputs MUST be:
 * - localFileUri: file://...
 * - frame: MLKit face frame
 */
export async function runAntiSpoofFromFrame(opts: {
  localFileUri: string
  frame: AnyMlKitFrame
  model: TensorflowModel
  threshold?: number
  returnDebug?: boolean
}): Promise<AntiSpoofResult> {
  const threshold = typeof opts.threshold === 'number' ? opts.threshold : 0.2

  const xywh = normalizeFrameToXywh(opts.frame)
  const square = makeSquareFrame(xywh)
  const face256Uri = await cropAndResize256(opts.localFileUri, square)

  const decoded = await decodeImageFileToRgba(face256Uri)
  if (decoded.width !== 256 || decoded.height !== 256) {
    throw new Error(`Expected 256x256 after crop+resize, got ${decoded.width}x${decoded.height}`)
  }

  const input = rgba256ToRgbFloatNHWC01(decoded.data)

  const outputs = await opts.model.run([input])
  const o0 = outputs?.[0]
  const o1 = outputs?.[1]
  if (!o0 || !o1) {
    throw new Error(`AntiSpoof model did not return 2 outputs.`)
  }

  const clssPred = o0 instanceof Float32Array ? o0 : new Float32Array(o0)
  const leafMask = o1 instanceof Float32Array ? o1 : new Float32Array(o1)

  const score = computeAntiSpoofScore(clssPred, leafMask)
  const isLive = score < threshold

  return {
    score,
    threshold,
    isLive,
    debug: opts.returnDebug ? { face256Uri } : undefined,
  }
}

/** Convenience: detect face + run anti-spoof */
export async function runAntiSpoofOnImage(opts: {
  localFileUri: string
  model: TensorflowModel
  threshold?: number
  returnDebug?: boolean
}): Promise<AntiSpoofResult> {
  const faces = await FaceDetection.detect(opts.localFileUri, {
    landmarkMode: 'all',
    trackingEnabled: false,
    performanceMode: 'fast',
  })
  if (!faces || faces.length === 0) throw new Error(`No face detected for anti-spoof: ${opts.localFileUri}`)
  const frame = faces[0]?.frame
  if (!frame) throw new Error('Face bbox(frame) missing for anti-spoof')

  return runAntiSpoofFromFrame({
    localFileUri: opts.localFileUri,
    frame: frame as AnyMlKitFrame,
    model: opts.model,
    threshold: opts.threshold,
    returnDebug: opts.returnDebug,
  })
}

/**
 * High-level API: detect face + check anti-spoof, returns true if live.
 */
export async function checkAntiSpoof(
  localFileUri: string,
  model: TensorflowModel,
  threshold?: number,
): Promise<{ isLive: boolean; score: number; threshold: number }> {
  const result = await runAntiSpoofOnImage({
    localFileUri,
    model,
    threshold,
  })
  return { isLive: result.isLive, score: result.score, threshold: result.threshold }
}
