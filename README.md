# Real-Time Face Recognition with Watchlist + Unknown Face Handling

A webcam stream detects faces in real time, tracks them across frames, extracts embeddings, compares against a gallery database, and shows:
- recognized person name
- confidence / similarity score
- track same person across frames
- 'unknown' if below threshold
- entry/exit event log
- simple dashboard or API for recent detections
- compare them with enrolled people using vector search

**Key elements**:
- computer vision pipeline design 
- similarity search
- real-time optimization
- backend engineering
- deployability

**Stack**:
- *InsightFace / ArcFace-style pipeline*: face detection + embeddings
  - InsightFace is widely used for face analysis and recognition pipelines
  - ArcFace: built into InsightFace, stable embeddings(angular margin loss)
- *FAISS*: identity search for nearest-neighbor lookup over face embeddings
  - FAISS is a fast way to find the closest embedding vectors and the standard practical tool for vector similarity search
- *OpenCV*: image / video / UI
  - The standart tool for image and video manipulation
- *FastAPI*: backend

## Architecture
Both parts work through common files with data:
- `data/face_db.npy`
- `data/events.jsonl`

Project has two entrypoints:

1. Camera (online)
> python main.py

2. API (offline)
> python -m app.api

### Online pipeline
> python main.py

1. Read frame from webcam
2. Detect faces
3. Crop / align face
4. Extract embedding
5. Search nearest neighbor in FAISS
6. Apply decision logic (threshold-based classification)
7. Render frame and save event

### Offline pipeline
> Load: uvicorn api:app --host localhost --port 8000 --reload
> python -m app.api

1. Data collection (add new person images)  
2. Preprocessing (crop / align faces)  
3. Embedding extraction  
4. Aggregation (average or store multiple embeddings)  
5. Index update (rebuild / update FAISS index)

UI:
- http://localhost:8000/ -> JSON
- http://localhost:8000/docs -> Swagger
- http://localhost:8000/redoc -> ReDoc

## Installation

```bash
git clone https://github.com/jane-kirina/real_time_face_recognition
cd real_time_face_recognition

pip install -r requirements.txt
python main.py
```

## Docker

Build image:

```bash
docker build -t face-recognition-app .

Run container:

docker run face-recognition-app
```

> Note: Camera access may require additional configuration depending on OS

## Plan of work
**Time limit: 1-2 weeks**

Goals:
- Clean pipeline design
- Real-time performance
- Clear README
- Understanding embeddings, trade-offs and optimizations


| Weeks/days   | Task                      | Details                                                                                                                          | Goal                               |
| ------------ | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| ***1 Week*** | *MVP*                     | *Add core system*                                                                                                                | *Create end-to-end system*         |
| Day 1        | Setup & Webcam            | Set up project structure, environment, webcam stream, frame rendering                                                            | working video loop                 |
| Day 2        | Face Detection            | Integrate face detection InsightFace, detect faces on each frame & draw bounding boxes                                           | faces highlighted in real time     |
| Day 3        | Embeddings                | extract embeddings for each detected face<br>test consistency: same face -> similar vectors                                      | understanding embeddings           |
| Day 4        | Enrollment System         | load images of a person & compute embeddings<br>save: vectors (numpy / pickle), labels (name -> embeddings)                      | small face database exists         |
| Day 5        | Recognition               | compare embeddings manually (cosine similarity), match against stored people<br>add threshold: above -> name, below -> 'unknown' | recognition works (simple version) |
| Day 6        | Real-time Recognition     | connect recognition to webcam pipeline, show name and confidence score                                                           | full pipeline working              |
| Day 7        | Refactor & Clean Code     | Write README, clean code, add docs                                                                                               | Make project looks structured      |
| ***2 Week*** |                           |                                                                                                                                  | *Add CV-level*                     |
| Day 8        | FAISS Integration         | replace manual search with FAISS<br>support: top-1 match, top-k (optional)                                                       | scalable recognition               |
| Day 9        | Tracking (reduce flicker) | simple tracking: assign ID per face, match by bbox proximity<br>reuse identity across frames                                     | labels stop jumping                |
| Day 10       | Temporal Smoothing        | store last N predictions per face<br>apply: majority vote or average confidence                                                  | stable predictions + optimizations |
| Day 11       | Logging System            | log events: person detected, unknown detected<br>avoid spam (cooldown per person)                                                | realistic system behavior          |
| Day 12       | FastAPI                   | Add minimal API:<br>- `POST /enroll`<br>- `GET /events`<br>- `GET /persons`                                                      | simple backend layer exists        |
| Day 13       | Benchmark & Testing       | measure: FPS, latency per frame<br>test: known faces, unknown faces<br>tune threshold                                            | visuals & numbers for README       |
| Day 14       | Final Day                 | Edit README(architecture, pipeline, decisions etc.)<br>Add screenshots / GIF, demo video<br>Optional Docker                      | ready to show project              |



### Progress logs

| Day | What has been done                                                                                                                                                                                                                                                                                                                                                                                                                   | Time |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| 1   | Added ReadMe with plan for project<br>Created webcam stream<br><br>3 types of fps:<br>- instant<br>- smoothing<br>- 1-second average                                                                                                                                                                                                                                                                                                 | 2.5h |
| 2   | Added face detection using InsightFace:<br>- camera works<br>- detector works<br>- rectangles are drawn                                                                                                                                                                                                                                                                                                                              | 2h   |
| 3   | *Completed tasks from 3-7 days*<br><br>Add detect + embedding<br>Research: embedding, FaceNet vs ArcFace<br>- saving embeddings to file<br>- comparing new embeddings to saved ones<br>- threshold for known / unknown face<br><br>Some optimizations<br><br>Added `handle_keypress` to webcam `start_camera()`<br>- exit<br>- save frame to folder `/outputs`<br>- pause frame + overlay text 'Paused'<br><br>Refactor & Clean Code | 4.5h |
| 4   | Sunday - weekday                                                                                                                                                                                                                                                                                                                                                                                                                     |      |
| 5   | *Completed tasks from 8-9 days*<br>Research: FAISS docs, tutorials<br>Fixed action_save()<br><br>FAISS Integration:<br>- replace manual search with FAISS, top-1 match<br><br>Tracking:<br>- simple tracking: assign ID per face, match by bbox proximity<br>- reuse identity across frames Goal: labels stop jumping                                                                                                                | 4h   |
| 6   | Some optimizations, fixed bugs<br><br>Add Logging System:<br>- save logs to json<br>- log events: person detected, unknown detected<br>- avoid spam (cooldown per person)<br><br>Added Offline pipeline, API logic:<br>- `GET /events`<br>- `GET /persons`<br>- `POST /enroll`<br>- `DELETE /persons/{name}`<br><br>Edit README(architecture, pipeline, instalation etc.)<br>Added Docker<br><br>Move state dict to class<br>        | 6h |
| 7   |                                                                                                                                                                                                                                                                                                                                                                                                                                      |      |






## Design Notes
This project focuses on building a clean and modular real-time face recognition pipeline:
- detection
- embedding extraction
- FAISS-based search
- tracking

To keep the architecture simple and readable, advanced optimizations such as:
- temporal smoothing
- detailed benchmarking and evaluation

were intentionally left out of the initial version. The current design allows these components to be added without major refactoring

Note: Threshold is currently fixed and not tuned on a validation set


