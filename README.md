# Real-Time Face Recognition with Watchlist + Unknown Face Handling

A webcam or RTSP stream detects faces in real time, tracks them across frames, extracts embeddings, compares against a gallery database, and shows:
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
- threshold tuning
- handling false positives
- real-time optimization
- backend engineering
- deployability

**Stack**:
- *InsightFace / ArcFace-style pipeline*: face detection + embeddings
- *ONNX Runtime*: fast inference runtime
- *FAISS*: identity search for nearest-neighbor lookup over face embeddings
- *OpenCV*: image / video / UI
- *FastAPI*: backend
- (optional, 3 week) *SQLite or PostgreSQL*: storage for metadata, local files for embeddings/images

*InsightFace is widely used for face analysis and recognition pipelines, and ONNX Runtime is a strong choice for deployment-focused inference. FAISS is the standard practical tool for vector similarity search*

## Architecture
### Online pipeline
1. Read frame from webcam
2. Detect faces
3. Crop / align face
4. Extract embedding
5. Search nearest neighbor in FAISS
6. Apply threshold:
    - above threshold -> known
    - below threshold -> unknown
7. Track person across frames
8. Smooth identity over several frames
9. Render frame and save event

### Offline pipeline
- add new person images
- compute embeddings
- average or store several embeddings
- rebuild / update FAISS index

## Plan of work
**Time limit: 1-2 weeks**



| Weeks/days   | Task                      | Details                                                                                                                          | Goal                               |
| ------------ | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| ***1 Week*** | *MVP*                     | *Add core system*                                                                                                                | *Create end-to-end system*         |
| ~~Day 1~~    | ~~Setup & Webcam~~        | ~~Set up project structure, environment, webcam stream, frame rendering~~                                                        | ~~working video loop~~             |
| ~~Day 2~~    | ~~Face Detection~~        | ~~Integrate face detection InsightFace, detect faces on each frame & draw bounding boxes~~                                       | ~~faces highlighted in real time~~ |
| Day 3        | Embeddings                | extract embeddings for each detected face<br>test consistency: same face -> similar vectors                                      | understanding embeddings           |
| Day 4        | Enrollment System         | load images of a person & compute embeddings<br>save: vectors (numpy / pickle), labels (name -> embeddings)                      | small face database exists         |
| Day 5        | Recognition               | compare embeddings manually (cosine similarity), match against stored people<br>add threshold: above -> name, below -> 'unknown' | recognition works (simple version) |
| Day 6        | Real-time Recognition     | connect recognition to webcam pipeline, show name and confidence score                                                           | full pipeline working              |
| Day 7        | Refactor & Clean Code     | Write README, create demo video/GIF, add benchmark section, add config file (threshold etc.)                                     | Make project looks structured      |
| ***2 Week*** |                           |                                                                                                                                  | *Add CV-level*                     |
| Day 8        | FAISS Integration         | replace manual search with FAISS<br>support: top-1 match, top-k (optional)                                                       | scalable recognition               |
| Day 9        | Tracking (reduce flicker) | simple tracking: assign ID per face, match by bbox proximity<br>reuse identity across frames                                     | labels stop jumping                |
| Day 10       | Temporal Smoothing        | store last N predictions per face<br>apply: majority vote or average confidence                                                  | stable predictions                 |
| Day 11       | Logging System            | log events: person detected, unknown detected<br>avoid spam (cooldown per person)                                                | realistic system behavior          |
| Day 12       | FastAPI                   | Add minimal API:<br>- `POST /enroll`<br>- `GET /events`<br>- `GET /persons`                                                      | simple backend layer exists        |
| Day 13       | Benchmark & Testing       | measure: FPS, latency per frame<br>test: known faces, unknown faces<br>tune threshold                                            | visuals & numbers for README       |
| Day 14       | Final Polish              | Polish README(architecture, pipeline, decisions etc.)<br>Add: screenshots / GIF, demo video<br>optional: Docker                  | ready to show project              |

---
 *Progress*

| Day | Status | What has been done                                                                                                                       | Time |
| --- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 1   | ✔      | Added ReadMe with plan for project<br><br>Created webcam stream<br><br>3 types of fps:<br>- instant<br>- smoothing<br>- 1-second average | 2.5h |
| 2   | ✔      | Added face detection using InsightFace:<br>- camera works<br>- detector works<br>- rectangles are drawn                                  | 2h   |
| 3   | ➖      |                                                                                                                                          |      |
| 4   | ➖      |                                                                                                                                          |      |
| 5   | ➖      |                                                                                                                                          |      |
| 6   | ➖      |                                                                                                                                          |      |
| 7   | ➖      |                                                                                                                                          |      |
| 8   | ➖      |                                                                                                                                          |      |
| 9   | ➖      |                                                                                                                                          |      |
| 10  | ➖      |                                                                                                                                          |      |
| 11  | ➖      |                                                                                                                                          |      |
| 12  | ➖      |                                                                                                                                          |      |
| 13  | ➖      |                                                                                                                                          |      |
| 14  | ➖      |                                                                                                                                          |      |

