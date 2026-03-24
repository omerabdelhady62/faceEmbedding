import React, { useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  Alert,
  Button,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  useColorScheme,
  View,
} from 'react-native'
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context'
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite'
import { Buffer } from 'buffer'
import { Picker } from '@react-native-picker/picker'

;(globalThis as any).Buffer = (globalThis as any).Buffer || Buffer

import {
  register,
  clockInOut,
  initDbs,
  fetchDb,
  deleteDb,
  type AppDbs,
  type DbName,
} from './face-recognition-module'

function nowMs(): number {
  const p = (globalThis as any)?.performance
  return typeof p?.now === 'function' ? p.now() : Date.now()
}

// --- App ---

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
  const [operation, setOperation] = useState<'clockin' | 'clockout'>('clockin')
  const [dbReady, setDbReady] = useState(false)

  const embedModelRef = useRef<TensorflowModel | null>(null)
  const antiSpoofModelRef = useRef<TensorflowModel | null>(null)
  const dbsRef = useRef<AppDbs | null>(null)

  const getEmbeddingModel = async (): Promise<TensorflowModel> => {
    if (embedModelRef.current) return embedModelRef.current
    embedModelRef.current = await loadTensorflowModel(require('./face-recognition-module/model/mobilefacenet.tflite'))
    return embedModelRef.current
  }

  const getAntiSpoofModel = async (): Promise<TensorflowModel> => {
    if (antiSpoofModelRef.current) return antiSpoofModelRef.current
    antiSpoofModelRef.current = await loadTensorflowModel(require('./face-recognition-module/model/FaceAntiSpoofing.tflite'))
    return antiSpoofModelRef.current
  }

  useEffect(() => {
    ;(async () => {
      try {
        dbsRef.current = await initDbs()
        setDbReady(true)
      } catch (e: any) {
        console.error(e)
        setOutput(`DB init error: ${e?.message ?? String(e)}`)
      }
    })()
  }, [])

  // --- Register flow ---
  const demoRegister = async () => {
    const source = require('./assets/mena.jpeg')
    try {
      setLoading(true)
      setOutput('')
      if (!dbsRef.current) throw new Error('DB not ready')
      const t0 = nowMs()
      const success = await register(source, employeeId, Date.now(), await getEmbeddingModel(), dbsRef.current)
      const t1 = nowMs()
      setOutput(
        success
          ? `Registered: ${employeeId.trim()}\nTime: ${(t1 - t0).toFixed(2)} ms`
          : `Registration failed for: ${employeeId.trim()}`,
      )
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  // --- Clock-in/out flow ---
  const demoClockInOut = async () => {
    const source = require('./assets/mena.jpeg')
    try {
      setLoading(true)
      setOutput('')
      if (!dbsRef.current) throw new Error('DB not ready')
      const t0 = nowMs()
      const res = await clockInOut(
        source,
        Date.now(),
        operation,
        await getEmbeddingModel(),
        await getAntiSpoofModel(),
        dbsRef.current,
        { matchThreshold: 0.6, liveThreshold: 0.2 },
      )
      const t1 = nowMs()

      const lines: string[] = [`Operation: ${operation}`]

      if (!res.embeddingsOk) {
        lines.push('Face detection FAILED (no face found)')
      } else if (!res.spoofOk) {
        lines.push('NOT LIVE (spoof detected)')
        lines.push(`Anti-spoof score: ${res.antiSpoofScore.toFixed(6)}`)
      } else if (res.matchedEmployeeId) {
        lines.push(`LIVE - Anti-spoof score: ${res.antiSpoofScore.toFixed(6)}`)
        lines.push(`MATCH: ${res.matchedEmployeeId}`)
        lines.push(`Confidence (cosine): ${res.matchScore.toFixed(6)}`)
      } else {
        lines.push(`LIVE - Anti-spoof score: ${res.antiSpoofScore.toFixed(6)}`)
        lines.push('NO MATCH (unknown employee)')
        if (res.matchScore >= 0) lines.push(`Best cosine: ${res.matchScore.toFixed(6)}`)
      }

      lines.push(`Time: ${(t1 - t0).toFixed(2)} ms`)
      setOutput(lines.join('\n'))
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    } finally {
      setLoading(false)
    }
  }

  // --- Utils ---
  const handleFetchDb = async (dbName: DbName) => {
    try {
      if (!dbsRef.current) throw new Error('DB not ready')
      const result = await fetchDb(dbsRef.current, dbName)
      if (result === -1) {
        setOutput(`No content in ${dbName} database`)
        return
      }
      if (dbName === 'employees') {
        const rows = result as any[]
        setOutput(
          rows
            .map((r) => `#${r.id}  ${r.employee_id}  (saved: ${new Date(r.created_at).toLocaleString()})`)
            .join('\n'),
        )
      } else {
        const rows = result as any[]
        setOutput(
          rows
            .map((r) => {
              const live = r.is_live === 1 ? 'LIVE' : 'SPOOF'
              const matched = r.matched === 1 ? `MATCH (${r.employee_id ?? 'UNKNOWN'})` : 'NO MATCH'
              return `#${r.id}  ${r.operation}  ${live}  spoof=${Number(r.anti_spoof_score).toFixed(4)}  ${matched}  cosine=${r.match_score == null ? '-' : Number(r.match_score).toFixed(4)}  (${new Date(r.created_at).toLocaleString()})`
            })
            .join('\n'),
        )
      }
    } catch (e: any) {
      console.error(e)
      setOutput(`Error: ${e?.message ?? String(e)}`)
    }
  }

  const handleDeleteDb = async (dbName: DbName) => {
    Alert.alert('Confirm', `Delete all data in ${dbName} database?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            if (!dbsRef.current) throw new Error('DB not ready')
            const ok = await deleteDb(dbsRef.current, dbName)
            setOutput(ok ? `Deleted all ${dbName} data` : `Failed to delete ${dbName} data`)
          } catch (e: any) {
            setOutput(`Error: ${e?.message ?? String(e)}`)
          }
        },
      },
    ])
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Face Recognition Demo</Text>

        {/* --- Registration --- */}
        <Text style={styles.sectionTitle}>Register Employee</Text>
        <Text style={styles.label}>Employee ID</Text>
        <TextInput
          value={employeeId}
          onChangeText={setEmployeeId}
          placeholder="e.g., 1001"
          style={styles.input}
          autoCapitalize="none"
        />
        <View style={styles.buttonWrap}>
          <Button
            title={dbReady ? 'Register (demo image)' : 'DB loading...'}
            onPress={demoRegister}
            disabled={loading || !dbReady}
          />
        </View>

        {/* --- Clock-in/out --- */}
        <Text style={styles.sectionTitle}>Clock-in / Clock-out</Text>
        <Text style={styles.label}>Operation</Text>
        <View style={styles.pickerWrap}>
          <Picker
            selectedValue={operation}
            onValueChange={(val: 'clockin' | 'clockout') => setOperation(val)}
            style={styles.picker}
          >
            <Picker.Item label="Clock In" value="clockin" />
            <Picker.Item label="Clock Out" value="clockout" />
          </Picker>
        </View>
        <View style={styles.buttonWrap}>
          <Button
            title={dbReady ? `${operation === 'clockin' ? 'Clock In' : 'Clock Out'} (demo image)` : 'DB loading...'}
            onPress={demoClockInOut}
            disabled={loading || !dbReady}
          />
        </View>

        {/* --- Utils --- */}
        <Text style={styles.sectionTitle}>Utils</Text>
        <View style={styles.row}>
          <View style={styles.rowBtn}>
            <Button title="Fetch Employees" onPress={() => handleFetchDb('employees')} disabled={loading || !dbReady} />
          </View>
          <View style={styles.rowBtn}>
            <Button title="Fetch Operations" onPress={() => handleFetchDb('operations')} disabled={loading || !dbReady} />
          </View>
        </View>
        <View style={styles.row}>
          <View style={styles.rowBtn}>
            <Button title="Delete Employees" onPress={() => handleDeleteDb('employees')} disabled={loading || !dbReady} color="#c00" />
          </View>
          <View style={styles.rowBtn}>
            <Button title="Delete Operations" onPress={() => handleDeleteDb('operations')} disabled={loading || !dbReady} color="#c00" />
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
  title: { fontSize: 18, fontWeight: '800', marginBottom: 16, color: '#111', lineHeight: 24 },
  sectionTitle: { fontSize: 14, fontWeight: '700', color: '#222', marginTop: 16, marginBottom: 8 },
  label: { fontSize: 12, fontWeight: '700', color: '#333', marginBottom: 4 },
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
  pickerWrap: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    marginBottom: 12,
    overflow: 'hidden',
  },
  picker: { height: 50 },
  buttonWrap: { marginTop: 4, marginBottom: 8 },
  row: { flexDirection: 'row', gap: 12, marginTop: 4, marginBottom: 4 },
  rowBtn: { flex: 1 },
  loader: { marginVertical: 12 },
  result: { marginTop: 12, fontSize: 12, fontWeight: '600', color: '#333', lineHeight: 18 },
})

export default App
