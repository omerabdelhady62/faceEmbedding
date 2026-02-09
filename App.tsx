import React, { useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  Alert,
  Button,
  Image,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  useColorScheme,
  View,
} from 'react-native'
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context'
import FaceDetection from '@react-native-ml-kit/face-detection'
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite'
import RNFS from 'react-native-fs'
import { Buffer } from 'buffer'

// SQLite (bare RN). If you are on Expo, swap this to `expo-sqlite`.
import SQLite from 'react-native-sqlite-storage'

  // ✅ IMPORTANT: polyfill Buffer for the environment
  ; (globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

import {
  getMobileFaceNetEmbeddingFromFrame,
  cropFace112ForDebug,
  type FaceFrame,
} from './faceEmbedding.ts'

SQLite.enablePromise(true)

function nowMs(): number {
  const p = (globalThis as any)?.performance
  return typeof p?.now === 'function' ? p.now() : Date.now()
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`Embedding length mismatch: ${a.length} vs ${b.length}`)
  let dot = 0,
    na = 0,
    nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i],
      y = b[i]
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

/**
 * Float32Array <-> base64 helpers for SQLite.
 * We store embeddings as base64 in TEXT column (portable across iOS/Android).
 */
function float32ToBase64(arr: Float32Array): string {
  const buf = Buffer.from(arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength))
  return buf.toString('base64')
}

function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, 'base64')
  // Node Buffer is backed by Uint8Array; create a copy aligned to 4 bytes
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  return new Float32Array(ab)
}

/**
 * ✅ FIX: Handles both remote URLs and local require() assets
 */
async function resolveImageToLocalUri(source: any, tag: string): Promise<string> {
  let uri: string

  if (typeof source === 'number') {
    // It's a local require() asset ID
    const resolved = Image.resolveAssetSource(source)
    uri = resolved.uri
  } else if (typeof source === 'string') {
    // It's a remote URL
    uri = source
  } else {
    throw new Error(`Unsupported image source type for ${tag}`)
  }

  const extension = uri.includes('.png') ? 'png' : 'jpg'
  const destPath = `${RNFS.TemporaryDirectoryPath}/img_${tag}_${Date.now()}.${extension}`

  // We use downloadFile even for local assets because in Dev mode
  // they are served over http from the Metro server.
  const res = await RNFS.downloadFile({
    fromUrl: uri,
    toFile: destPath,
  }).promise

  if (res.statusCode && res.statusCode >= 400) {
    throw new Error(`Failed to resolve image ${tag} (status ${res.statusCode})`)
  }

  return `file://${destPath}`
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
 * DB schema
 */
const DB_NAME = 'employees.db'
const TABLE_SQL = `
CREATE TABLE IF NOT EXISTS employees (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  embedding_b64 TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
`

type EmployeeRow = { id: number; name: string; embedding_b64: string; created_at: number }

async function openDb() {
  return SQLite.openDatabase({ name: DB_NAME, location: 'default' })
}

async function initDb() {
  const db = await openDb()
  await db.executeSql(TABLE_SQL)
  return db
}

async function upsertEmployee(db: any, name: string, embedding: Float32Array) {
  const embedding_b64 = float32ToBase64(embedding)
  const created_at = Date.now()
  // Insert or replace on name uniqueness
  await db.executeSql(
    `INSERT INTO employees(name, embedding_b64, created_at)
     VALUES(?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET embedding_b64=excluded.embedding_b64, created_at=excluded.created_at;`,
    [name.trim(), embedding_b64, created_at],
  )
}

async function getAllEmployees(db: any): Promise<EmployeeRow[]> {
  const [res] = await db.executeSql(`SELECT id, name, embedding_b64, created_at FROM employees ORDER BY id ASC;`)
  const rows: EmployeeRow[] = []
  for (let i = 0; i < res.rows.length; i++) rows.push(res.rows.item(i))
  return rows
}

async function deleteAllEmployees(db: any) {
  await db.executeSql(`DELETE FROM employees;`)
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
  const [employeeName, setEmployeeName] = useState('')
  const [dbReady, setDbReady] = useState(false)

  const modelRef = useRef<TensorflowModel | null>(null)
  const dbRef = useRef<any>(null)

  const getModel = async (): Promise<TensorflowModel> => {
    if (modelRef.current) return modelRef.current
    modelRef.current = await loadTensorflowModel(require('./model/mobilefacenet.tflite'))
    return modelRef.current
  }

  useEffect(() => {
    ; (async () => {
      try {
        dbRef.current = await initDb()
        setDbReady(true)
      } catch (e: any) {
        console.error(e)
        setOutput(`DB init error: ${e?.message ?? String(e)}`)
      }
    })()
  }, [])

  /**
   * 1) New employee registration
   * - input: name + image
   * - output: embedding saved to SQLite
   */
  const registerEmployee = async (name: string, imageSource: any) => {
    const trimmed = name.trim()
    if (!trimmed) throw new Error('Employee name is required')
    if (!dbRef.current) throw new Error('DB not ready')

    const model = await getModel()

    const img = await resolveImageToLocalUri(imageSource, `reg_${trimmed}`)
    const frame = await detectFirstFaceFrame(img)

    // Optional: debug crop
    try {
      const crop = await cropFace112ForDebug(img, frame)
      if (crop) await saveFileUriToDocuments(crop, `reg_crop_${trimmed}_${Date.now()}.jpg`)
    } catch {
      // ignore
    }

    const emb = await getMobileFaceNetEmbeddingFromFrame(img, frame, model)
    await upsertEmployee(dbRef.current, trimmed, emb)

    return { name: trimmed, embeddingLength: emb.length }
  }

  /**
   * 2) Employee clock-in
   * - input: image
   * - output: best matched name + confidence score
   */
  const clockIn = async (imageSource: any, threshold = 0.6) => {
    if (!dbRef.current) throw new Error('DB not ready')

    const model = await getModel()

    const img = await resolveImageToLocalUri(imageSource, `clockin_${Date.now()}`)
    const frame = await detectFirstFaceFrame(img)

    const probe = await getMobileFaceNetEmbeddingFromFrame(img, frame, model)

    const employees = await getAllEmployees(dbRef.current)
    if (employees.length === 0) throw new Error('No employees registered in DB')

    let bestName: string | null = null
    let bestScore = -1

    for (const emp of employees) {
      const galleryEmb = base64ToFloat32(emp.embedding_b64)
      const score = cosineSimilarity(probe, galleryEmb)
      if (score > bestScore) {
        bestScore = score
        bestName = emp.name
      }
    }

    const matched = bestScore >= threshold
    return {
      matched,
      name: matched ? bestName : null,
      score: bestScore,
      threshold,
      comparedAgainst: employees.length,
    }
  }

  // --- Demo UI wiring (swap the imageSource for camera frames in your real app) ---
  const demoRegister = async () => {
    // Replace with an image picker / camera capture in your app
    const source = require('./assets/mena.jpeg')
    try {
      setLoading(true)
      setOutput('')
      const t0 = nowMs()
      const res = await registerEmployee(employeeName, source)
      const t1 = nowMs()
      setOutput(
        [
          `✅ Registered: ${res.name}`,
          `Embedding length: ${res.embeddingLength}`,
          `Time: ${(t1 - t0).toFixed(2)} ms`,
        ].join('\n'),
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  const demoClockIn = async () => {
    // Replace with an image picker / camera capture in your app
    const source = require('./assets/ana.jpeg')
    const THRESHOLD = 0.6

    try {
      setLoading(true)
      setOutput('')
      const t0 = nowMs()
      const res = await clockIn(source, THRESHOLD)
      const t1 = nowMs()

      setOutput(
        [
          `Best match: ${res.name ?? 'UNKNOWN'}`,
          `Confidence (cosine): ${res.score.toFixed(6)}`,
          `Threshold: ${res.threshold.toFixed(2)}`,
          `Decision: ${res.matched ? '✅ MATCH' : '❌ NO MATCH'}`,
          `Compared against: ${res.comparedAgainst} employees`,
          `Time: ${(t1 - t0).toFixed(2)} ms`,
        ].join('\n'),
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  const showAllEmployees = async () => {
    try {
      if (!dbRef.current) throw new Error('DB not ready')
      const rows = await getAllEmployees(dbRef.current)
      setOutput(
        rows.length === 0
          ? 'No employees in DB'
          : rows
            .map((r) => `• #${r.id}  ${r.name}  (saved: ${new Date(r.created_at).toLocaleString()})`)
            .join('\n'),
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    }
  }

  const clearDb = async () => {
    try {
      if (!dbRef.current) throw new Error('DB not ready')
      await deleteAllEmployees(dbRef.current)
      setOutput('✅ Deleted all employees')
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Offline Face Recognition (SQLite)</Text>

        <Text style={styles.label}>Employee name (for registration)</Text>
        <TextInput
          value={employeeName}
          onChangeText={setEmployeeName}
          placeholder="e.g., Omar"
          style={styles.input}
          autoCapitalize="words"
        />

        <View style={styles.buttonWrap}>
          <Button title={dbReady ? '1) Register employee (demo image)' : 'DB loading...'} onPress={demoRegister} disabled={loading || !dbReady} />
        </View>
        <View style={styles.buttonWrap}>
          <Button title={dbReady ? '2) Clock-in (demo image)' : 'DB loading...'} onPress={demoClockIn} disabled={loading || !dbReady} />
        </View>

        <View style={styles.row}>
          <View style={styles.rowBtn}>
            <Button title="List employees" onPress={showAllEmployees} disabled={loading || !dbReady} />
          </View>
          <View style={styles.rowBtn}>
            <Button
              title="Clear DB"
              onPress={() =>
                Alert.alert('Confirm', 'Delete all employees?', [
                  { text: 'Cancel', style: 'cancel' },
                  { text: 'Delete', style: 'destructive', onPress: clearDb },
                ])
              }
              disabled={loading || !dbReady}
            />
          </View>
        </View>

        {loading && <ActivityIndicator size="small" style={styles.loader} />}
        {!!output && <Text style={styles.result}>{output}</Text>}

        <Text style={styles.note}>
          Notes:\n• Registration stores a single embedding per employee name (upsert).\n• Clock-in computes cosine similarity against all stored embeddings and returns the best match + score.\n• Replace the demo `require(...)` images with a real camera/picker image URI.
        </Text>
      </ScrollView>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 24 },
  title: { fontSize: 18, fontWeight: '800', marginBottom: 16, color: '#111' },
  label: { fontSize: 12, fontWeight: '700', color: '#333', marginBottom: 8 },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 12,
    fontSize: 14,
    color: '#111',
  },
  buttonWrap: { marginTop: 8, marginBottom: 8 },
  row: { flexDirection: 'row', gap: 12, marginTop: 8, marginBottom: 8 },
  rowBtn: { flex: 1 },
  loader: { marginVertical: 12 },
  result: { marginTop: 12, fontSize: 12, fontWeight: '600', color: '#333', lineHeight: 18 },
  note: { marginTop: 16, fontSize: 12, color: '#444', lineHeight: 18 },
})

export default App
