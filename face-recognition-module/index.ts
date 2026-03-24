// Main flows
export { register, clockInOut, type ClockInOutResult } from './main'

// Utils
export { resolveImageToLocalUri, fetchDb, deleteDb, getTime, type DbName } from './utils'

// DB
export { initDbs, createEmployeesDb, type AppDbs, type EmployeeRow, type OperationRow, type EmbeddingEntry } from './db'

// Lower-level (for advanced usage)
export { getEmbeddings, matchEmployee } from './faceEmbedding'
export { checkAntiSpoof } from './antiSpoof'
