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
;(globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

import {
  getMobileFaceNetEmbeddingFromFrame,
  cropFace112ForDebug,
  type FaceFrame as FaceFrameTLWH,
} from './faceEmbedding'

import { runAntiSpoofFromFrame, type AnyMlKitFrame } from './antiSpoof'

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
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  return new Float32Array(ab)
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

  // AntiSpoof JS decoders don't support webp; so keep assets jpg/png.
  const extension = uri.toLowerCase().includes('.png') ? 'png' : 'jpg'
  const destPath = `${RNFS.TemporaryDirectoryPath}/img_${tag}_${Date.now()}.${extension}`

  // We use downloadFile even for local assets because in dev mode they are served over http from Metro.
  const res = await RNFS.downloadFile({ fromUrl: uri, toFile: destPath }).promise

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

function toTLWH(frame: AnyMlKitFrame): FaceFrameTLWH {
  const anyF: any = frame
  // MLKit sometimes returns x/y, sometimes left/top (depending on wrapper versions)
  const left = typeof anyF.left === 'number' ? anyF.left : anyF.x
  const top = typeof anyF.top === 'number' ? anyF.top : anyF.y
  if (![left, top, anyF.width, anyF.height].every((v) => typeof v === 'number')) {
    throw new Error(`Invalid face frame shape: ${JSON.stringify(frame)}`)
  }
  return { left, top, width: anyF.width, height: anyF.height }
}

async function detectFirstFaceFrame(localFileUri: string): Promise<AnyMlKitFrame> {
  const faces = await FaceDetection.detect(localFileUri, {
    landmarkMode: 'all',
    trackingEnabled: false,
    performanceMode: 'fast',
  })
  if (!faces || faces.length === 0) throw new Error(`No face detected for: ${localFileUri}`)
  const frame = faces[0]?.frame
  if (!frame) throw new Error(`Face bbox(frame) missing for: ${localFileUri}`)
  return frame as AnyMlKitFrame
}

/**
 * DB schema
 */
const DB_NAME = 'employees.db'

const EMPLOYEES_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS employees (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id TEXT NOT NULL UNIQUE,
  embedding_b64 TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
`

const CLOCKINS_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS clockins (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id TEXT,
  is_live INTEGER NOT NULL,
  anti_spoof_score REAL NOT NULL,
  matched INTEGER NOT NULL,
  match_score REAL,
  match_threshold REAL,
  created_at INTEGER NOT NULL
);
`

type EmployeeRow = { id: number; employee_id: string; embedding_b64: string; created_at: number }
type ClockInRow = { id: number; employee_id: string | null; is_live: number; anti_spoof_score: number; matched: number; match_score: number | null; match_threshold: number | null; created_at: number }

async function openDb() {
  return SQLite.openDatabase({ name: DB_NAME, location: 'default' })
}

async function initDb() {
  const db = await openDb()
  await db.executeSql(EMPLOYEES_TABLE_SQL)
  await db.executeSql(CLOCKINS_TABLE_SQL)
  return db
}

async function upsertEmployee(db: any, employeeId: string, embedding: Float32Array) {
  const embedding_b64 = float32ToBase64(embedding)
  const created_at = Date.now()
  await db.executeSql(
    `INSERT INTO employees(employee_id, embedding_b64, created_at)
     VALUES(?, ?, ?)
     ON CONFLICT(employee_id) DO UPDATE SET embedding_b64=excluded.embedding_b64, created_at=excluded.created_at;`,
    [employeeId.trim(), embedding_b64, created_at],
  )
}

async function getAllEmployees(db: any): Promise<EmployeeRow[]> {
  const [res] = await db.executeSql(
    `SELECT id, employee_id, embedding_b64, created_at FROM employees ORDER BY id ASC;`,
  )
  const rows: EmployeeRow[] = []
  for (let i = 0; i < res.rows.length; i++) rows.push(res.rows.item(i))
  return rows
}

async function deleteAllEmployees(db: any) {
  await db.executeSql(`DELETE FROM employees;`)
}


async function insertClockin(
  db: any,
  row: {
    employee_id: string | null
    is_live: boolean
    anti_spoof_score: number
    matched: boolean
    match_score: number | null
    match_threshold: number | null
  },
) {
  const created_at = Date.now()
  await db.executeSql(
    `INSERT INTO clockins (employee_id, is_live, anti_spoof_score, matched, match_score, match_threshold, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?);`,
    [
      row.employee_id ?? null,
      row.is_live ? 1 : 0,
      row.anti_spoof_score,
      row.matched ? 1 : 0,
      row.match_score ?? null,
      row.match_threshold ?? null,
      created_at,
    ],
  )
}

async function getAllClockins(db: any, limit = 50): Promise<ClockInRow[]> {
  const [res] = await db.executeSql(
    `SELECT id, employee_id, is_live, anti_spoof_score, matched, match_score, match_threshold, created_at
     FROM clockins
     ORDER BY id DESC
     LIMIT ?;`,
    [limit],
  )
  const rows: ClockInRow[] = []
  for (let i = 0; i < res.rows.length; i++) rows.push(res.rows.item(i))
  return rows
}

async function deleteAllClockins(db: any) {
  await db.executeSql(`DELETE FROM clockins;`)
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
  const [employeeId, setEmployeeId] = useState('')
  const [dbReady, setDbReady] = useState(false)

  const embedModelRef = useRef<TensorflowModel | null>(null)
  const antiSpoofModelRef = useRef<TensorflowModel | null>(null)
  const dbRef = useRef<any>(null)

  const getEmbeddingModel = async (): Promise<TensorflowModel> => {
    if (embedModelRef.current) return embedModelRef.current
    embedModelRef.current = await loadTensorflowModel(require('./model/mobilefacenet.tflite'))
    return embedModelRef.current
  }

  const getAntiSpoofModel = async (): Promise<TensorflowModel> => {
    if (antiSpoofModelRef.current) return antiSpoofModelRef.current
    antiSpoofModelRef.current = await loadTensorflowModel(require('./model/FaceAntiSpoofing.tflite'))
    return antiSpoofModelRef.current
  }

  useEffect(() => {
    ;(async () => {
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
   * 1) Registration
   * - input: image + employeeId
   * - output: embedding saved to SQLite
   */
  const registerEmployee = async (employeeIdRaw: string, imageSource: any) => {
    const trimmed = employeeIdRaw.trim()
    if (!trimmed) throw new Error('Employee ID is required')
    if (!dbRef.current) throw new Error('DB not ready')

    const embedModel = await getEmbeddingModel()

    const img = await resolveImageToLocalUri(imageSource, `reg_${trimmed}`)
    const frameAny = await detectFirstFaceFrame(img)
    const frameTLWH = toTLWH(frameAny)

    // Optional: debug crop (112x112) for MobileFaceNet
    try {
      const crop = await cropFace112ForDebug(img, frameTLWH)
      if (crop) await saveFileUriToDocuments(crop, `reg_crop112_${trimmed}_${Date.now()}.jpg`)
    } catch {
      // ignore
    }

    const emb = await getMobileFaceNetEmbeddingFromFrame(img, frameTLWH, embedModel)
    await upsertEmployee(dbRef.current, trimmed, emb)

    return { employeeId: trimmed, embeddingLength: emb.length }
  }

  /**
   * 2) Clock-in
   * - input: image
   * - pipeline:
   *   (a) detect face
   *   (b) anti-spoof (256x256 preprocessing + score)
   *   (c) if live: compute embedding (112x112 preprocessing)
   *   (d) compare against all employees in SQLite, return best match
   */
  const clockIn = async (imageSource: any, opts?: { matchThreshold?: number; liveThreshold?: number }) => {
    if (!dbRef.current) throw new Error('DB not ready')

    const matchThreshold = opts?.matchThreshold ?? 0.6
    const liveThreshold = opts?.liveThreshold ?? 0.01

    const embedModel = await getEmbeddingModel()
    const antiSpoofModel = await getAntiSpoofModel()

    const img = await resolveImageToLocalUri(imageSource, `clockin_${Date.now()}`)
    const frameAny = await detectFirstFaceFrame(img)

    const live = await runAntiSpoofFromFrame({
      localFileUri: img,
      frame: frameAny,
      model: antiSpoofModel,
      threshold: liveThreshold,
      returnDebug: true,
    })

    if (!live.isLive) {
      await insertClockin(dbRef.current, {
        employee_id: null,
        is_live: false,
        anti_spoof_score: live.score,
        matched: false,
        match_score: null,
        match_threshold: matchThreshold,
      })
      return {
        ok: false as const,
        reason: 'SPOOF' as const,
        antiSpoofScore: live.score,
        liveThreshold: live.threshold,
        debug: live.debug,
      }
    }

    const frameTLWH = toTLWH(frameAny)
    const probe = await getMobileFaceNetEmbeddingFromFrame(img, frameTLWH, embedModel)

    const employees = await getAllEmployees(dbRef.current)
    if (employees.length === 0) throw new Error('No employees registered in DB')

    let bestEmployeeId: string | null = null
    let bestScore = -1

    for (const emp of employees) {
      const galleryEmb = base64ToFloat32(emp.embedding_b64)
      const score = cosineSimilarity(probe, galleryEmb)
      if (score > bestScore) {
        bestScore = score
        bestEmployeeId = emp.employee_id
      }
    }

    const matched = bestScore >= matchThreshold

    await insertClockin(dbRef.current, {
      employee_id: matched ? bestEmployeeId : null,
      is_live: true,
      anti_spoof_score: live.score,
      matched,
      match_score: bestScore,
      match_threshold: matchThreshold,
    })

    return {
      ok: true as const,
      antiSpoofScore: live.score,
      liveThreshold: live.threshold,
      matched,
      employeeId: matched ? bestEmployeeId : null,
      matchScore: bestScore,
      matchThreshold,
      comparedAgainst: employees.length,
      debug: live.debug,
    }
  }

  // --- Demo UI wiring (swap the imageSource for camera frames in your real app) ---
  const demoRegister = async () => {
    const source = require('./assets/ana.jpeg')
    try {
      setLoading(true)
      setOutput('')
      const t0 = nowMs()
      const res = await registerEmployee(employeeId, source)
      const t1 = nowMs()
      setOutput([`✅ Registered: ${res.employeeId}`, `Embedding length: ${res.embeddingLength}`, `Time: ${(t1 - t0).toFixed(2)} ms`].join('\n'))
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  const demoClockIn = async () => {
    const source = require('./assets/ana.jpeg')
    const MATCH_THRESHOLD = 0.6
    const LIVE_THRESHOLD = 0.2

    try {
      setLoading(true)
      setOutput('')
      const t0 = nowMs()
      const res = await clockIn(source, { matchThreshold: MATCH_THRESHOLD, liveThreshold: LIVE_THRESHOLD })
      const t1 = nowMs()

      if (!res.ok) {
        setOutput(
          [
            `❌ NOT LIVE (spoof detected)`,
            `Anti-spoof score: ${res.antiSpoofScore.toFixed(6)} (threshold ${res.liveThreshold})`,
            res.debug?.face256Uri ? `Debug face256: ${res.debug.face256Uri}` : '',
            `Time: ${(t1 - t0).toFixed(2)} ms`,
          ]
            .filter(Boolean)
            .join('\n'),
        )
        return
      }

      setOutput(
        [
          `✅ LIVE`,
          `Anti-spoof score: ${res.antiSpoofScore.toFixed(6)} (threshold ${res.liveThreshold})`,
          '',
          `Best match: ${res.employeeId ?? 'UNKNOWN'}`,
          `Confidence (cosine): ${res.matchScore.toFixed(6)}`,
          `Threshold: ${res.matchThreshold.toFixed(2)}`,
          `Decision: ${res.matched ? '✅ MATCH' : '❌ NO MATCH'}`,
          `Compared against: ${res.comparedAgainst} employees`,
          '',
          res.debug?.face256Uri ? `Debug face256: ${res.debug.face256Uri}` : '',
          `Time: ${(t1 - t0).toFixed(2)} ms`,
        ]
          .filter(Boolean)
          .join('\n'),
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
              .map((r) => `• #${r.id}  ${r.employee_id}  (saved: ${new Date(r.created_at).toLocaleString()})`)
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

  const showClockins = async () => {
    try {
      if (!dbRef.current) throw new Error('DB not ready')
      const rows = await getAllClockins(dbRef.current, 50)
      setOutput(
        rows.length === 0
          ? 'No clock-ins yet'
          : rows
              .map((r) => {
                const live = r.is_live === 1 ? 'LIVE' : 'SPOOF'
                const matched = r.matched === 1 ? `MATCH (${r.employee_id ?? 'UNKNOWN'})` : 'NO MATCH'
                return `• #${r.id}  ${live}  spoof=${Number(r.anti_spoof_score).toFixed(6)}  ${matched}  cosine=${r.match_score == null ? '—' : Number(r.match_score).toFixed(6)}  (${new Date(r.created_at).toLocaleString()})`
              })
              .join('\n'),
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    }
  }

  const clearClockins = async () => {
    try {
      if (!dbRef.current) throw new Error('DB not ready')
      await deleteAllClockins(dbRef.current)
      setOutput('✅ Deleted all clock-ins')
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    }
  }


  return (
    <View style={[styles.container, { paddingTop: insets.top }]}> 
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Employee Clock-in Demo (Anti-Spoof + Face Embeddings + SQLite)</Text>

        <Text style={styles.label}>Employee ID (for registration)</Text>
        <TextInput
          value={employeeId}
          onChangeText={setEmployeeId}
          placeholder="e.g., 1001"
          style={styles.input}
          autoCapitalize="none"
        />

        <View style={styles.buttonWrap}>
          <Button
            title={dbReady ? '1) Register (demo image)' : 'DB loading...'}
            onPress={demoRegister}
            disabled={loading || !dbReady}
          />
        </View>

        <View style={styles.buttonWrap}>
          <Button
            title={dbReady ? '2) Clock-in (demo image)' : 'DB loading...'}
            onPress={demoClockIn}
            disabled={loading || !dbReady}
          />
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
          
        <View style={styles.row}>
          <View style={styles.rowBtn}>
            <Button title="List clock-ins" onPress={showClockins} disabled={loading || !dbReady} />
          </View>
          <View style={styles.rowBtn}>
            <Button
              title="Clear clock-ins"
              onPress={() =>
                Alert.alert('Confirm', 'Delete all clock-ins?', [
                  { text: 'Cancel', style: 'cancel' },
                  { text: 'Delete', style: 'destructive', onPress: clearClockins },
                ])
              }
              disabled={loading || !dbReady}
            />
          </View>
        </View>

</View>
        </View>

        {loading && <ActivityIndicator size="small" style={styles.loader} />}
        {!!output && <Text style={styles.result}>{output}</Text>}


      </ScrollView>
    </View>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 24 },
  title: { fontSize: 16, fontWeight: '800', marginBottom: 16, color: '#111', lineHeight: 22 },
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
