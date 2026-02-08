import ImageEditor from '@react-native-community/image-editor';
import RNFS from 'react-native-fs';
import { Skia } from '@shopify/react-native-skia';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { Buffer } from 'buffer';

export type FaceFrame = {
  top: number;
  left: number;
  width: number;
  height: number;
};

/**
 * Crop + resize to 112x112 from a LOCAL file:// image URI.
 * Uses @react-native-community/image-editor (replacement for deprecated core ImageEditor).
 */
async function cropFace112(
  localFileUri: string,
  frame: FaceFrame,
): Promise<string> {
  if (!localFileUri.startsWith('file://')) {
    throw new Error(
      `imageUri must be a local file:// URI. Got: ${localFileUri}`,
    );
  }

  const cropData = {
    offset: { x: Math.max(0, frame.left), y: Math.max(0, frame.top) },
    size: {
      width: Math.max(1, frame.width),
      height: Math.max(1, frame.height),
    },
    displaySize: { width: 112, height: 112 },
    resizeMode: 'cover' as const,
  };

  // ✅ FIX: Modern ImageEditor returns an object { uri: string, width: number, ... }
  const result = await ImageEditor.cropImage(localFileUri, cropData);

  const croppedUri = typeof result === 'string' ? result : result.uri;

  // Ensure the URI has the file:// prefix for RNFS/Skia
  if (!croppedUri.startsWith('file://') && !croppedUri.startsWith('http')) {
    return `file://${croppedUri}`;
  }

  return croppedUri;
}

/**
 * Decode a 112x112 cropped image into Float32 input tensor [1,112,112,3].
 * Assumes croppedUri is file:// so RNFS can read.
 */
async function image112ToInputTensor(
  croppedUri: string,
): Promise<Float32Array> {
  if (!croppedUri || typeof croppedUri !== 'string') {
    throw new Error(
      'image112ToInputTensor: croppedUri is undefined or not a string',
    );
  }

  const path = croppedUri.replace('file://', '');
  const exists = await RNFS.exists(path);
  if (!exists) throw new Error(`File does not exist at path: ${path}`);

  const base64 = await RNFS.readFile(path, 'base64');
  const bytes = new Uint8Array(Buffer.from(base64, 'base64'));

  const skData = Skia.Data.fromBytes(bytes);
  const skImage = Skia.Image.MakeImageFromEncoded(skData);

  if (!skImage)
    throw new Error('Failed to decode cropped face image into Skia.');

  // ✅ FIX: Ensure offsets (0,0) and dimensions (112,112) are explicitly passed
  // Some Skia versions require the full ImageInfo object
  const pixels = skImage.readPixels(0, 0, {
    width: 112,
    height: 112,
    colorType: 4, // RGBA_8888
    alphaType: 1, // Opaque
  });

  if (!pixels)
    throw new Error(
      'Failed to read pixels. Check if image dimensions are exactly 112x112.',
    );

  // pixels.length should be 112 * 112 * 4 = 50176
  const input = new Float32Array(1 * 112 * 112 * 3);
  let j = 0;

  for (let i = 0; i < pixels.length; i += 4) {
    // Only process if we haven't overfilled the input array
    if (j >= input.length) break;

    const r = pixels[i + 0];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    // Normalization: (x - 127.5) / 128.0
    input[j++] = (r - 127.5) / 128.0;
    input[j++] = (g - 127.5) / 128.0;
    input[j++] = (b - 127.5) / 128.0;
  }

  return input;
}

function l2Normalize(vec: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

/**
 * Main API:
 * - local file:// image
 * - MLKit bbox frame
 * - MobileFaceNet TFLite model (react-native-fast-tflite)
 * -> returns L2-normalized embedding (Float32Array)
 */
export async function getMobileFaceNetEmbeddingFromFrame(
  localFileUri: string,
  frame: FaceFrame,
  model: TensorflowModel,
): Promise<Float32Array> {
  const cropped112Uri = await cropFace112(localFileUri, frame);
  const input = await image112ToInputTensor(cropped112Uri);

  const outputs = model.runSync([input]);
  const embedding = outputs[0] as Float32Array;

  return l2Normalize(embedding);
}

/**
 * Optional helper: get the cropped face URI (for saving / debugging).
 */
export async function cropFace112ForDebug(
  localFileUri: string,
  frame: FaceFrame,
): Promise<string> {
  return cropFace112(localFileUri, frame);
}
