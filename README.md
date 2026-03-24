# Face Recognition Demo

Offline face registration + clock-in/clock-out with anti-spoof detection.
All processing runs on-device using TFLite models (React Native).

---

## 1. How It Works

```
                         ┌──────────────────────────┐
                         │       INPUT: Image        │
                         └────────────┬─────────────┘
                                      │
                              ┌───────▼───────┐
                              │  MLKit Face   │
                              │  Detection    │
                              └───────┬───────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
            ┌───────▼───────┐                     ┌───────▼────────┐
            │   REGISTER    │                     │  CLOCK-IN/OUT  │
            └───────┬───────┘                     └───────┬────────┘
                    │                                     │
          ┌─────────▼──────────┐              ┌───────────▼───────────┐
          │  Get Embedding     │              │  Get Embedding        │
          │  (MobileFaceNet)   │              │  (MobileFaceNet)      │
          │  112x112 cover     │              │  112x112 cover        │
          │  (x-127.5)/128     │              │  (x-127.5)/128        │
          └─────────┬──────────┘              └───────────┬───────────┘
                    │                                     │
          ┌─────────▼──────────┐              ┌───────────▼───────────┐
          │  Save to           │              │  Anti-Spoof Check     │
          │  employees.db      │              │  256x256 stretch      │
          └─────────┬──────────┘              │  x/255 normalize      │
                    │                         └───────────┬───────────┘
                    ▼                                     │
               return                           ┌────────▼────────┐
               true/false                       │  score < 0.2 ?  │
                                                └──┬──────────┬───┘
                                              NO   │          │  YES (LIVE)
                                                   ▼          │
                                             log spoof   ┌────▼──────────┐
                                                         │ Match against │
                                                         │ all employees │
                                                         │ cosine >= 0.6 │
                                                         └────┬──────────┘
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │  Save operation    │
                                                    │  to operations.db  │
                                                    └─────────┬──────────┘
                                                              │
                                                              ▼
                                                       return result
```

---

## 2. Project Structure

```
App.tsx                                 ← UI only (demo buttons + output)
│
└── face-recognition-module/            ← SELF-CONTAINED — copy this folder to integrate
    │
    ├── index.ts                        Barrel exports (import everything from here)
    ├── main.ts                         register() + clockInOut()
    ├── utils.ts                        resolveImageToLocalUri() + fetchDb() + deleteDb()
    ├── faceEmbedding.ts                Embedding pipeline + matchEmployee()
    ├── antiSpoof.ts                    Anti-spoof pipeline + checkAntiSpoof()
    ├── db.ts                           SQLite operations (2 databases)
    │
    └── model/
        ├── mobilefacenet.tflite        Embedding model  [1,112,112,3]
        └── FaceAntiSpoofing.tflite     Anti-spoof model [1,256,256,3]
```

---

## 3. Integration Guide

Everything you need is exported from `./face-recognition-module`.

### 3.1 Setup

```ts
import { initDbs, type AppDbs } from './face-recognition-module'
import { loadTensorflowModel } from 'react-native-fast-tflite'

// 1. Initialize databases (call once at app start)
const dbs: AppDbs = await initDbs()

// 2. Load models (call once, cache the references)
const embedModel = await loadTensorflowModel(require('./face-recognition-module/model/mobilefacenet.tflite'))
const antiSpoofModel = await loadTensorflowModel(require('./face-recognition-module/model/FaceAntiSpoofing.tflite'))
```

---

### 3.2 `register()`

Register an employee's face.

```ts
import { register } from './face-recognition-module'

const success = await register(imageSource, employeeId, time, embedModel, dbs)
```

```
┌─────────────────────────────────────────────────────────┐
│  register()                                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUTS                                                 │
│  ──────                                                 │
│  imageSource    any         require('./photo.jpg')       │
│                             or 'file:///path/to/img'    │
│  employeeId     string      e.g. '1001'                 │
│  time           number      Date.now() (ms timestamp)   │
│  embedModel     TFLiteModel loaded MobileFaceNet model  │
│  dbs            AppDbs      from initDbs()              │
│                                                         │
│  OUTPUT                                                 │
│  ──────                                                 │
│  Promise<boolean>                                       │
│    true  → saved successfully                           │
│    false → save failed                                  │
│                                                         │
│  THROWS                                                 │
│  ──────                                                 │
│  - 'Employee ID is required' (empty string)             │
│  - 'No face detected' (no face in image)                │
│                                                         │
│  WHAT IT DOES                                           │
│  ────────────                                           │
│  image → face detect → embedding → save to employees.db │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3.3 `clockInOut()`

Perform a clock-in or clock-out operation.

```ts
import { clockInOut } from './face-recognition-module'

const result = await clockInOut(
  imageSource,             // image
  Date.now(),              // time
  'clockin',               // or 'clockout'
  embedModel,              // loaded model
  antiSpoofModel,          // loaded model
  dbs,                     // from initDbs()
  { matchThreshold: 0.6, liveThreshold: 0.2 },  // optional
)
```

```
┌───────────────────────────────────────────────────────────────┐
│  clockInOut()                                                 │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUTS                                                       │
│  ──────                                                       │
│  imageSource      any                  image to process       │
│  time             number               Date.now()             │
│  operation        'clockin'|'clockout' operation type         │
│  embedModel       TFLiteModel          MobileFaceNet model    │
│  antiSpoofModel   TFLiteModel          anti-spoof model       │
│  dbs              AppDbs               from initDbs()         │
│  opts? (optional)                                             │
│    matchThreshold  number              default 0.6            │
│    liveThreshold   number              default 0.2            │
│                                                               │
│  OUTPUT                                                       │
│  ──────                                                       │
│  Promise<ClockInOutResult>                                    │
│  {                                                            │
│    embeddingsOk: boolean         // face detection succeeded  │
│    spoofOk: boolean              // liveness check passed     │
│    matchedEmployeeId: string|null// matched ID or null        │
│    antiSpoofScore: number        // raw spoof score           │
│    matchScore: number            // cosine similarity         │
│  }                                                            │
│                                                               │
│  RESULT INTERPRETATION                                        │
│  ─────────────────────                                        │
│  embeddingsOk=false              → no face found in image     │
│  embeddingsOk=true, spoofOk=false → spoof detected            │
│  spoofOk=true, matchedEmployeeId=null → live but no match     │
│  spoofOk=true, matchedEmployeeId='1001' → success!            │
│                                                               │
│  NOTE: every attempt is auto-saved to operations.db           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### 3.4 `fetchDb()`

Retrieve all records from a database.

```ts
import { fetchDb } from './face-recognition-module'

const result = await fetchDb(dbs, 'employees')   // or 'operations'
```

```
┌────────────────────────────────────────────────────────────────┐
│  fetchDb()                                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  INPUTS                                                        │
│  ──────                                                        │
│  dbs        AppDbs              from initDbs()                 │
│  dbName     'employees' | 'operations'                         │
│                                                                │
│  OUTPUT                                                        │
│  ──────                                                        │
│  Promise< EmployeeRow[] | OperationRow[] | -1 >                │
│                                                                │
│    -1  → database is empty (no records)                        │
│                                                                │
│    if dbName='employees' → EmployeeRow[]                       │
│    ┌──────────────────────────────────────────┐                │
│    │ { id, employee_id, embedding_b64,        │                │
│    │   created_at }                           │                │
│    └──────────────────────────────────────────┘                │
│                                                                │
│    if dbName='operations' → OperationRow[]                     │
│    ┌──────────────────────────────────────────┐                │
│    │ { id, employee_id, operation, is_live,   │                │
│    │   anti_spoof_score, matched, match_score,│                │
│    │   match_threshold, created_at }          │                │
│    └──────────────────────────────────────────┘                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

### 3.5 `deleteDb()`

Delete all records from a database.

```ts
import { deleteDb } from './face-recognition-module'

const success = await deleteDb(dbs, 'employees')   // or 'operations'
```

```
┌─────────────────────────────────────────────────┐
│  deleteDb()                                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  INPUTS                                         │
│  ──────                                         │
│  dbs        AppDbs              from initDbs()  │
│  dbName     'employees' | 'operations'          │
│                                                 │
│  OUTPUT                                         │
│  ──────                                         │
│  Promise<boolean>                               │
│    true  → deleted successfully                 │
│    false → deletion failed                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 3.6 Quick Reference

```
FUNCTION          INPUT                                  OUTPUT
─────────────     ─────────────────────────────────────  ──────────────────────────
initDbs()         (none)                                 AppDbs
register()        image, employeeId, time, model, dbs    boolean
clockInOut()      image, time, op, 2 models, dbs, opts?  ClockInOutResult
fetchDb()         dbs, dbName                            rows[] | -1
deleteDb()        dbs, dbName                            boolean
```

---

## 4. Databases

Two separate SQLite databases:

### `employees.db`

```sql
CREATE TABLE employees (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id    TEXT NOT NULL UNIQUE,     -- e.g. '1001'
  embedding_b64  TEXT NOT NULL,            -- Base64-encoded Float32Array
  created_at     INTEGER NOT NULL          -- ms timestamp
);
```

### `operations.db`

```sql
CREATE TABLE operations (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  employee_id      TEXT,                   -- matched ID or NULL
  operation        TEXT NOT NULL,          -- 'clockin' | 'clockout'
  is_live          INTEGER NOT NULL,       -- 1=live, 0=spoof
  anti_spoof_score REAL NOT NULL,          -- raw score
  matched          INTEGER NOT NULL,       -- 1=matched, 0=no match
  match_score      REAL,                   -- cosine similarity
  match_threshold  REAL,                   -- threshold used
  created_at       INTEGER NOT NULL        -- ms timestamp
);
```

Every clock-in/out attempt is logged, including spoof detections and unmatched faces.

---

## 5. Models

```
MODEL               INPUT                NORMALIZATION       RESIZE        OUTPUT
──────────────────  ───────────────────  ──────────────────  ────────────  ─────────────────
MobileFaceNet       float32[1,112,112,3] (pixel-127.5)/128  112x112 cover L2-norm embedding
FaceAntiSpoofing    float32[1,256,256,3] pixel/255           256x256 stretch score (LIVE<0.2)
```

> **Warning:** These two pipelines use DIFFERENT preprocessing (normalization, resize mode, image decoder). They must NOT be mixed.

---

## 6. Tech Stack

- React Native 0.83 + TypeScript
- `react-native-fast-tflite` (TFLite inference)
- `@react-native-ml-kit/face-detection` (face detection)
- `@shopify/react-native-skia` (image decode for embeddings)
- `jpeg-js` / `pngjs` (image decode for anti-spoof)
- `react-native-sqlite-storage` (SQLite)
- `@react-native-community/image-editor` (crop/resize)
- `@react-native-picker/picker` (operation type selection)

---

## 7. How to Run

**Prerequisites:** Node 20+, JDK 17, Android Studio, `JAVA_HOME` and `ANDROID_HOME` set.

```bash
npm install
npm start            # Metro bundler
npm run android      # Run on Android
```

| Command | Description |
|---|---|
| `npm start` | Start Metro bundler |
| `npm run android` | Run on Android |
| `npm run lint` | ESLint |
| `npm test` | Jest tests |

---

## 8. Known Issues

- `jcenter()` in `react-native-sqlite-storage` needs replacing with `mavenCentral()`
- Anti-spoof JS decoders don't support webp -- use jpg/png for input images
