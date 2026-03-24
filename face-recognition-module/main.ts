import type { TensorflowModel } from 'react-native-fast-tflite'
import { getEmbeddings, matchEmployee } from './faceEmbedding'
import { checkAntiSpoof } from './antiSpoof'
import { saveEmbedding, saveOperation, getAllEmployees, type AppDbs } from './db'
import { resolveImageToLocalUri } from './utils'

export type ClockInOutResult = {
  embeddingsOk: boolean
  spoofOk: boolean
  matchedEmployeeId: string | null
  antiSpoofScore: number
  matchScore: number
}

export async function register(
  imageSource: any,
  employeeId: string,
  time: number,
  embedModel: TensorflowModel,
  dbs: AppDbs,
): Promise<boolean> {
  const trimmed = employeeId.trim()
  if (!trimmed) throw new Error('Employee ID is required')

  const img = await resolveImageToLocalUri(imageSource, `reg_${trimmed}`)
  const embedding = await getEmbeddings(img, embedModel)
  const saved = await saveEmbedding(dbs.employeesDb, trimmed, embedding, time)
  // TODO: save_embeddings_to_cloud() --> omar
  return saved
}

export async function clockInOut(
  imageSource: any,
  time: number,
  operation: 'clockin' | 'clockout',
  embedModel: TensorflowModel,
  antiSpoofModel: TensorflowModel,
  dbs: AppDbs,
  opts?: { matchThreshold?: number; liveThreshold?: number },
): Promise<ClockInOutResult> {
  const matchThreshold = opts?.matchThreshold ?? 0.6
  const liveThreshold = opts?.liveThreshold ?? 0.2

  const img = await resolveImageToLocalUri(imageSource, `op_${operation}_${Date.now()}`)

  // 1. Get embeddings
  let embedding: Float32Array
  try {
    embedding = await getEmbeddings(img, embedModel)
  } catch {
    await saveOperation(dbs.operationsDb, {
      employee_id: null,
      operation,
      is_live: false,
      anti_spoof_score: -1,
      matched: false,
      match_score: null,
      match_threshold: matchThreshold,
      time,
    })
    return { embeddingsOk: false, spoofOk: false, matchedEmployeeId: null, antiSpoofScore: -1, matchScore: -1 }
  }

  // 2. Check anti-spoof
  const spoof = await checkAntiSpoof(img, antiSpoofModel, liveThreshold)

  if (!spoof.isLive) {
    await saveOperation(dbs.operationsDb, {
      employee_id: null,
      operation,
      is_live: false,
      anti_spoof_score: spoof.score,
      matched: false,
      match_score: null,
      match_threshold: matchThreshold,
      time,
    })
    return { embeddingsOk: true, spoofOk: false, matchedEmployeeId: null, antiSpoofScore: spoof.score, matchScore: -1 }
  }

  // 3. Match employee
  const employees = await getAllEmployees(dbs.employeesDb)
  if (employees.length === 0) {
    await saveOperation(dbs.operationsDb, {
      employee_id: null,
      operation,
      is_live: true,
      anti_spoof_score: spoof.score,
      matched: false,
      match_score: null,
      match_threshold: matchThreshold,
      time,
    })
    return { embeddingsOk: true, spoofOk: true, matchedEmployeeId: null, antiSpoofScore: spoof.score, matchScore: -1 }
  }

  const match = matchEmployee(embedding, employees, matchThreshold)

  // 4. Save operation
  await saveOperation(dbs.operationsDb, {
    employee_id: match.employeeId,
    operation,
    is_live: true,
    anti_spoof_score: spoof.score,
    matched: match.matched,
    match_score: match.score,
    match_threshold: matchThreshold,
    time,
  })

  return {
    embeddingsOk: true,
    spoofOk: true,
    matchedEmployeeId: match.employeeId,
    antiSpoofScore: spoof.score,
    matchScore: match.score,
  }
}
