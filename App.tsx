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
  createEmployeesDb,
  type AppDbs,
  type DbName,
  type EmbeddingEntry,
} from './face-recognition-module'

function nowMs(): number {
  const p = (globalThis as any)?.performance
  return typeof p?.now === 'function' ? p.now() : Date.now()
}

function formatTimeDiff(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
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
  const [employeeName, setEmployeeName] = useState('')
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
      const success = await register(source, employeeId, employeeName, await getEmbeddingModel(), dbsRef.current)
      const t1 = nowMs()
      setOutput(
        success
          ? `Registered: ${employeeId.trim()} - ${employeeName.trim()}\nTime: ${(t1 - t0).toFixed(2)} ms`
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
        lines.push(
          `MATCH: ${res.matchedEmployeeId}${res.matchedEmployeeName ? ` - ${res.matchedEmployeeName}` : ''}`,
        )
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

  // --- Test createEmployeesDb ---
  const demoCreateEmployeesDb = async () => {
    try {
      setLoading(true)
      setOutput('')
      // const EMBED_SIZE = 192
      // const entries: EmbeddingEntry[] = [
      //   { employee_id: 'EMP001', embedding: Float32Array.from({ length: EMBED_SIZE }, (_, i) => Math.sin(i * 0.1)) },
      //   { employee_id: 'EMP002', embedding: Float32Array.from({ length: EMBED_SIZE }, (_, i) => Math.cos(i * 0.1)) },
      //   { employee_id: 'EMP003', embedding: Float32Array.from({ length: EMBED_SIZE }, (_, i) => Math.sin(i * 0.2 + 1)) },
      //   { employee_id: 'EMP004', embedding: Float32Array.from({ length: EMBED_SIZE }, (_, i) => Math.cos(i * 0.2 + 1)) },
      // ]
    const entries: EmbeddingEntry[] = [
        {
    employee_id: "69cbe418e8ce0577f8278c00",
    embedding:Float32Array.from( [
      -0.0015500792,
      0.01253964,
      -0.008538847,
      0.0006451485,
      -0.0052178735,
      -0.06745018,
      -0.019153796,
      0.26218638,
      0.020670898,
      -0.08711924,
      -0.009701881,
      -0.0058525805,
      0.004905904,
      -0.00662194,
      0.0017446314,
      -0.00062485883,
      -0.019725092,
      0.00023275081,
      -0.0012747974,
      0.014825142,
      -0.018502763,
      -0.04622005,
      0.16319153,
      0.0050161756,
      -0.06763218,
      -0.015610468,
      -0.008234993,
      -0.13287662,
      0.071108535,
      0.07645596,
      0.009394827,
      0.22524439,
      0.01506006,
      0.00018484723,
      0.062323675,
      0.076284796,
      0.116910905,
      0.02947629,
      -0.005142543,
      -0.04733629,
      0.008247777,
      -0.0014470431,
      0.003865168,
      -0.0021380396,
      0.004678268,
      0.007896302,
      0.01930897,
      -0.02161089,
      0.014846997,
      0.029573187,
      0.050922535,
      -0.00090793945,
      -0.113487296,
      0.0016380806,
      -0.07753811,
      0.006803944,
      -0.20861815,
      0.0017747029,
      -0.009129679,
      -0.011811173,
      -0.01642882,
      0.0155624505,
      -0.09742563,
      0.03815024,
      -0.012850467,
      -0.23232804,
      0.0021769586,
      0.013156879,
      0.008130944,
      -0.000050172574,
      -0.0036031597,
      -0.2426815,
      0.05244423,
      0.010868204,
      0.034146443,
      -0.015700078,
      0.0032598723,
      0.0016658329,
      0.10685359,
      0.1339485,
      -0.000638701,
      0.03184817,
      0.00094813004,
      -0.092865415,
      0.17911722,
      0.0025331376,
      -0.001772771,
      0.08135336,
      -0.054283526,
      0.13005322,
      -0.000099008495,
      -0.0005335378,
      0.0009925268,
      0.0050428794,
      -0.31121066,
      -0.15568626,
      0.042574927,
      -0.0875466,
      -0.004051912,
      0.043004915,
      -0.00036088983,
      0.0009014315,
      0.0058125914,
      -0.005819072,
      0.011355951,
      0.0032073562,
      0.039031282,
      -0.0026993204,
      0.008241057,
      0.003136816,
      0.10981322,
      0.008718693,
      0.004545552,
      -0.1779465,
      0.004348088,
      -0.01602563,
      -0.001145692,
      -0.012709044,
      0.013635689,
      0.08746161,
      0.023311617,
      -0.017183322,
      0.16524708,
      -0.0033810274,
      0.0022002964,
      0.0026016932,
      -0.005466157,
      0.0071568517,
      -0.011715121,
      -0.037491444,
      0.0041204174,
      0.006517848,
      0.0008922517,
      0.009710372,
      -0.07549554,
      -0.004447402,
      0.09453545,
      -0.089689136,
      0.013827985,
      -0.013988648,
      0.005459284,
      -0.009477636,
      -0.007389799,
      0.11780365,
      -0.09059398,
      -0.26086286,
      -0.005000057,
      -0.0039515533,
      -0.002712123,
      -0.0053985924,
      -0.015266337,
      -0.03309927,
      0.09840424,
      0.005638824,
      -0.0038673934,
      -0.0032309403,
      0.012303794,
      -0.013590534,
      0.04982793,
      0.002179162,
      0.011064828,
      0.000026508724,
      0.015555271,
      0.0021645932,
      -0.00014981668,
      -0.0039579887,
      -0.010829519,
      0.04708852,
      -0.00024044035,
      -0.006535105,
      -0.15495683,
      -0.052276306,
      -0.0023705969,
      -0.065597154,
      0.02227175,
      0.003377504,
      -0.0021568472,
      0.015229301,
      -0.000019895297,
      -0.00288859,
      0.15047614,
      -0.044096027,
      0.0025812315,
      0.00044806543,
      0.1508767,
      -0.06270592,
      0.0033505587,
      -0.013327917,
      -0.21763289,
      0.077397086,
      -0.0049802335,
      0.00007124447
    ])
    },]



      const ok = await createEmployeesDb(entries)
      if (ok) {
        setOutput(`Created employees DB with ${entries.length} dummy entries:\n${entries.map(e => e.employee_id).join(', ')}`)
      } else {
        setOutput('Failed to create employees DB')
      }
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
            .map((r) => `#${r.id}  ${r.employee_id} - ${r.employee_name}  (saved: ${new Date(r.created_at).toLocaleString()})`)
            .join('\n'),
        )
      } else {
        const rows = result as any[]
        setOutput(
          rows
            .map((r) => {
              const live = r.is_live === 1 ? 'LIVE' : 'SPOOF'
              const matched =
                r.matched === 1
                  ? `MATCH (${r.employee_id ?? 'UNKNOWN'}${r.employee_name ? ` - ${r.employee_name}` : ''})`
                  : 'NO MATCH'
              const ago = r.time_diff_seconds != null ? formatTimeDiff(r.time_diff_seconds) : ''
              return `#${r.id}  ${r.operation}  ${live}  spoof=${Number(r.anti_spoof_score).toFixed(4)}  ${matched}  cosine=${r.match_score == null ? '-' : Number(r.match_score).toFixed(4)}  ${ago}`
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
        <Text style={styles.label}>Employee Name</Text>
        <TextInput
          value={employeeName}
          onChangeText={setEmployeeName}
          placeholder="e.g., Mena Adel"
          style={styles.input}
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

        {/* --- Test createEmployeesDb --- */}
        <Text style={styles.sectionTitle}>Test DB</Text>
        <View style={styles.buttonWrap}>
          <Button
            title="Create Dummy Employees DB (4 entries)"
            onPress={demoCreateEmployeesDb}
            disabled={loading}
            color="#666"
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
