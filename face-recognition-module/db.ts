import SQLite from 'react-native-sqlite-storage'
import { Buffer } from 'buffer'

SQLite.enablePromise(true)

const EMPLOYEES_DB_NAME = 'employees.db'
const OPERATIONS_DB_NAME = 'operations.db'

const EMPLOYEES_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS employees (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id TEXT NOT NULL UNIQUE,
  embedding_b64 TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
`

const OPERATIONS_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS operations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id TEXT,
  operation TEXT NOT NULL,
  is_live INTEGER NOT NULL,
  anti_spoof_score REAL NOT NULL,
  matched INTEGER NOT NULL,
  match_score REAL,
  match_threshold REAL,
  created_at INTEGER NOT NULL
);
`

export type EmployeeRow = {
  id: number
  employee_id: string
  embedding_b64: string
  created_at: number
}

export type OperationRow = {
  id: number
  employee_id: string | null
  operation: string
  is_live: number
  anti_spoof_score: number
  matched: number
  match_score: number | null
  match_threshold: number | null
  created_at: number
}

export type AppDbs = { employeesDb: any; operationsDb: any }

// --- Float32Array <-> base64 helpers ---

export function float32ToBase64(arr: Float32Array): string {
  const buf = Buffer.from(arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength))
  return buf.toString('base64')
}

export function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, 'base64')
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  return new Float32Array(ab)
}

// --- Init ---

export async function initDbs(): Promise<AppDbs> {
  const employeesDb = await SQLite.openDatabase({ name: EMPLOYEES_DB_NAME, location: 'default' })
  await employeesDb.executeSql(EMPLOYEES_TABLE_SQL)
  const operationsDb = await SQLite.openDatabase({ name: OPERATIONS_DB_NAME, location: 'default' })
  await operationsDb.executeSql(OPERATIONS_TABLE_SQL)
  return { employeesDb, operationsDb }
}

// --- Employees DB ---

export async function saveEmbedding(
  db: any,
  employeeId: string,
  embedding: Float32Array,
  time: number,
): Promise<boolean> {
  try {
    const embedding_b64 = float32ToBase64(embedding)
    await db.executeSql(
      `INSERT INTO employees(employee_id, embedding_b64, created_at)
       VALUES(?, ?, ?)
       ON CONFLICT(employee_id) DO UPDATE SET embedding_b64=excluded.embedding_b64, created_at=excluded.created_at;`,
      [employeeId.trim(), embedding_b64, time],
    )
    return true
  } catch {
    return false
  }
}

export async function getAllEmployees(db: any): Promise<EmployeeRow[]> {
  const [res] = await db.executeSql(
    `SELECT id, employee_id, embedding_b64, created_at FROM employees ORDER BY id ASC;`,
  )
  const rows: EmployeeRow[] = []
  for (let i = 0; i < res.rows.length; i++) rows.push(res.rows.item(i))
  return rows
}

// --- Operations DB ---

export async function saveOperation(
  db: any,
  row: {
    employee_id: string | null
    operation: string
    is_live: boolean
    anti_spoof_score: number
    matched: boolean
    match_score: number | null
    match_threshold: number | null
    time: number
  },
): Promise<boolean> {
  try {
    await db.executeSql(
      `INSERT INTO operations (employee_id, operation, is_live, anti_spoof_score, matched, match_score, match_threshold, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?);`,
      [
        row.employee_id ?? null,
        row.operation,
        row.is_live ? 1 : 0,
        row.anti_spoof_score,
        row.matched ? 1 : 0,
        row.match_score ?? null,
        row.match_threshold ?? null,
        row.time,
      ],
    )
    return true
  } catch {
    return false
  }
}

export async function getAllOperations(db: any, limit = 50): Promise<OperationRow[]> {
  const [res] = await db.executeSql(
    `SELECT id, employee_id, operation, is_live, anti_spoof_score, matched, match_score, match_threshold, created_at
     FROM operations
     ORDER BY id DESC
     LIMIT ?;`,
    [limit],
  )
  const rows: OperationRow[] = []
  for (let i = 0; i < res.rows.length; i++) rows.push(res.rows.item(i))
  return rows
}

// --- Utils ---

export type DbName = 'employees' | 'operations'

function dbNameToSqliteName(dbName: DbName): string {
  return dbName === 'employees' ? EMPLOYEES_DB_NAME : OPERATIONS_DB_NAME
}

export async function getDbFile(dbName: DbName): Promise<any> {
  const name = dbNameToSqliteName(dbName)
  return SQLite.openDatabase({ name, location: 'default' })
}

export async function fetchDb(
  dbs: AppDbs,
  dbName: DbName,
): Promise<EmployeeRow[] | OperationRow[] | -1> {
  if (dbName === 'employees') {
    const rows = await getAllEmployees(dbs.employeesDb)
    return rows.length === 0 ? -1 : rows
  } else {
    const rows = await getAllOperations(dbs.operationsDb)
    return rows.length === 0 ? -1 : rows
  }
}

export async function deleteDb(dbs: AppDbs, dbName: DbName): Promise<boolean> {
  try {
    if (dbName === 'employees') {
      await dbs.employeesDb.executeSql(`DELETE FROM employees;`)
    } else {
      await dbs.operationsDb.executeSql(`DELETE FROM operations;`)
    }
    return true
  } catch {
    return false
  }
}
