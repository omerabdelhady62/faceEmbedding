# Employee Face Registration & Clock-In Demo

Offline Face Recognition + Anti-Spoof (React Native)

------------------------------------------------------------------------

## Overview

This demo mobile app implements three core biometric capabilities:

### 1) Registration

Registers an employee by: - Detecting face - Extracting face embedding -
Saving embedding in SQLite DB with employee ID

### 2) Clock-In

Authenticates employee by: - Detecting face - Running anti-spoof
(liveness) model - If LIVE → generate embedding - Compare with
registered embeddings - Return best matched employee ID

------------------------------------------------------------------------

# Architecture

Image\
↓\
MLKit Face Detection\
↓

Clock-In Flow: - AntiSpoof Model (256x256 stretch) - If LIVE → Face
Embedding Model (112x112 cover) - Cosine Matching - Save clock-in record

------------------------------------------------------------------------

# Models Used

## Face Embedding Model

-   MobileFaceNet
-   Input: float32\[1,112,112,3\]
-   Normalization: (pixel - 127.5) / 128
-   Output: 128D or 192D embedding

## Anti-Spoof Model

-   Input: float32\[1,256,256,3\]
-   Normalization: pixel / 255
-   Output:
    -   Identity: float32\[1,8\]
    -   Identity_1: float32\[1,8\]

LIVE condition: score \< 0.01

------------------------------------------------------------------------

# Preprocessing Pipelines

⚠ Anti-spoof and embedding pipelines are different and must not be
mixed.

## Embedding Preprocessing (faceEmbedding.ts)

1.  Detect face via MLKit
2.  Crop bounding box
3.  Resize to 112x112 (cover)
4.  Decode via Skia
5.  Normalize: (pixel - 127.5) / 128
6.  Create tensor \[1,112,112,3\]
7.  Run TFLite
8.  L2 normalize embedding

------------------------------------------------------------------------

## Anti-Spoof Preprocessing (antiSpoof.ts)

1.  Detect face
2.  Make bounding box square
3.  Crop
4.  Resize to 256x256 (stretch)
5.  Decode via jpeg-js or pngjs
6.  Normalize: pixel / 255
7.  Create tensor \[1,256,256,3\]
8.  Run TFLite
9.  Compute score
10. LIVE if score \< 0.01

------------------------------------------------------------------------

# Database Structure

SQLite DB: employees.db

## Table: employees

CREATE TABLE employees ( id INTEGER PRIMARY KEY AUTOINCREMENT,
employee_id TEXT, embedding TEXT );

Embedding stored as Base64 encoded Float32Array.

------------------------------------------------------------------------

## Table: clockins

CREATE TABLE clockins ( id INTEGER PRIMARY KEY AUTOINCREMENT,
employee_id TEXT, is_live INTEGER, anti_spoof_score REAL, matched
INTEGER, match_score REAL, match_threshold REAL, created_at TEXT );

Fields explanation:

-   employee_id: matched employee ID (NULL if no match)
-   is_live: 1 = live, 0 = spoof
-   anti_spoof_score: raw liveness score
-   matched: 1 = matched above threshold
-   match_score: cosine similarity score
-   match_threshold: threshold used for match
-   created_at: timestamp of attempt

Every clock-in attempt is logged, including spoof attempts.

------------------------------------------------------------------------

# Main Integration Functions

## registerEmployee

Signature: async function registerEmployee(imageUri: string, employeeId:
string)

Flow: - Detect face - Generate embedding - Save to SQLite

Usage: await registerEmployee("file:///path/face.jpg", "1001")

------------------------------------------------------------------------

## clockIn

Signature: async function clockIn(imageUri: string)

Returns: { employeeId: string \| null, score: number, isLive: boolean }

Flow: - Detect face - Run anti-spoof - Log spoof if detected - If live →
generate embedding - Compare embeddings - Log result - Return best match

Usage: const result = await clockIn(imageUri)

------------------------------------------------------------------------

## listClockIns

Signature: async function listClockIns()

Returns all clock-in records ordered by newest first.

------------------------------------------------------------------------

## clearClockIns

Signature: async function clearClockIns()

Deletes all clock-in records.

------------------------------------------------------------------------

# Matching Logic

Cosine similarity:

dot(a,b) / (\|\|a\|\| \* \|\|b\|\|)

Recommended threshold: \> 0.6 = match

------------------------------------------------------------------------

# How to Run (Windows)

1)  Install:

-   Node 20+
-   JDK 17
-   Android Studio
-   Set JAVA_HOME
-   Set ANDROID_HOME

2)  Install dependencies: npm install

3)  Start Metro: npm start

4)  Run Android: npx react-native run-android


------------------------------------------------------------------------

# Important Files

App.tsx faceEmbedding.ts antiSpoof.ts database.ts
models/mobilefacenet.tflite models/FaceAntiSpoofing.tflite

------------------------------------------------------------------------

# Summary

This demo provides fully offline: - Face registration - Anti-spoof
detection - Face recognition - SQLite embedding storage - Persistent
clock-in history

Integration entry points:

registerEmployee(imageUri, employeeId) clockIn(imageUri) listClockIns()
clearClockIns()
