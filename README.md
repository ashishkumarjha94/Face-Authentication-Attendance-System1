# Face Authentication Attendance System (Webcam)

## Overview
A local webcam-based face authentication attendance system:
- Enroll user face template (embedding)
- Identify face in real-time
- Mark attendance: Punch-in / Punch-out
- Handles lighting variation via CLAHE
- Basic liveness (spoof prevention) via challenge-response:
  - Blink OR head turn left/right using MediaPipe FaceMesh

## Model and Approach Used
- InsightFace (buffalo_l):
  - Face detection: RetinaFace
  - Face embedding: ArcFace-style 512D embedding (pretrained)
- Matching: cosine similarity with threshold.

Why this approach:
- High-quality pretrained embeddings avoid training from scratch
- Simple enrollment: store template embedding per user
- Works in real-time on CPU for a prototype

## Training Process
No full training is performed.
Enrollment creates a user template by:
1. capturing ~20 embeddings after liveness passes,
2. averaging embeddings,
3. L2-normalizing,
4. storing in SQLite.

(Optionally, you could build a per-user classifier later, but embeddings are enough for attendance.)

## How to Run

### Install
```bash
pip install -r requirements.txt