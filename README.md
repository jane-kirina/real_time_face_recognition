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

- *InsightFace is widely used for face analysis and recognition pipelines*
- *ArcFace: built into InsightFace, stable embeddings(angular margin loss)*
- *ONNX Runtime is a strong choice for deployment-focused inference*
- *FAISS is a fast way to find the closest embedding vectors and the standard practical tool for vector similarity search*

## Plan of work
**Time limit: 1-2 weeks**

Goals:
- Clean pipeline design
- Correct threshold handling
- Real-time performance
- Clear README + demo
- Understanding embeddings, trade-offs and optimizations


| Weeks/days   | Task                      | Details                                                                                                                              | Goal                                   |
| ------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| ***1 Week*** | *MVP*                     | *Add core system*                                                                                                                    | *Create end-to-end system*             |
| ~~Day 1~~    | ~~Setup & Webcam~~        | ~~Set up project structure, environment, webcam stream, frame rendering~~                                                            | ~~working video loop~~                 |
| ~~Day 2~~    | ~~Face Detection~~        | ~~Integrate face detection InsightFace, detect faces on each frame & draw bounding boxes~~                                           | ~~faces highlighted in real time~~     |
| ~~Day 3~~    | ~~Embeddings~~            | ~~extract embeddings for each detected face<br>test consistency: same face -> similar vectors~~                                      | ~~understanding embeddings~~           |
| ~~Day 4~~    | ~~Enrollment System~~     | ~~load images of a person & compute embeddings<br>save: vectors (numpy / pickle), labels (name -> embeddings)~~                      | ~~small face database exists~~         |
| ~~Day 5~~    | ~~Recognition~~           | ~~compare embeddings manually (cosine similarity), match against stored people<br>add threshold: above -> name, below -> 'unknown'~~ | ~~recognition works (simple version)~~ |
| ~~Day 6~~    | ~~Real-time Recognition~~ | ~~connect recognition to webcam pipeline, show name and confidence score~~                                                           | ~~full pipeline working~~              |
| ~~Day 7~~    | ~~Refactor & Clean Code~~ | ~~Write README, clean code, add docs~~                                                                                               | ~~Make project looks structured~~      |
| ***2 Week*** |                           |                                                                                                                                      | *Add CV-level*                         |
| ~~Day 8~~    | ~~FAISS Integration~~     | ~~replace manual search with FAISS<br>support: top-1 match, top-k (optional)~~                                                       | ~~scalable recognition~~               |
| ~~Day 9~~        | ~~Tracking (reduce flicker)~~ | ~~simple tracking: assign ID per face, match by bbox proximity<br>reuse identity across frames~~                                         | ~~labels stop jumping~~                    |
| Day 10       | Temporal Smoothing        | store last N predictions per face<br>apply: majority vote or average confidence                                                      | stable predictions                     |
| Day 11       | Logging System            | log events: person detected, unknown detected<br>avoid spam (cooldown per person)                                                    | realistic system behavior              |
| Day 12       | FastAPI                   | Add minimal API:<br>- `POST /enroll`<br>- `GET /events`<br>- `GET /persons`                                                          | simple backend layer exists            |
| Day 13       | Benchmark & Testing       | measure: FPS, latency per frame<br>test: known faces, unknown faces<br>tune threshold                                                | visuals & numbers for README           |
| Day 14       | Final Day                 | Edit README(architecture, pipeline, decisions etc.)<br>Add screenshots / GIF, demo video<br>Optional Docker                          | ready to show project                  |






### Progress logs



| Day | Status | What has been done                                                                                                                                                                                                                                                                                                                                                                                                 | Time |
| --- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| 1   | ✔      | Added ReadMe with plan for project<br>Created webcam stream<br>3 types of fps:<br>- instant<br>- smoothing<br>- 1-second average                                                                                                                                                                                                                                                                                   | 2.5h |
| 2   | ✔      | Added face detection using InsightFace:<br>- camera works<br>- detector works<br>- rectangles are drawn                                                                                                                                                                                                                                                                                                            | 2h   |
| 3   | ✔      | *Completed tasks from 3-7 days*<br>Add detect + embedding<br>Research: embedding, FaceNet vs ArcFace<br>saving embeddings to file<br>comparing new embeddings to saved ones<br>threshold for known / unknown face<br>Some optimizations<br>Added `handle_keypress` to webcam `start_camera()`<br>- exit<br>- save frame to folder `/outputs`<br>- pause frame + overlay text 'Paused'<br><br>Refactor & Clean Code | 4.5h |
| 4   | ➖      | Sunday - weekday                                                                                                                                                                                                                                                                                                                                                                                                   |      |
| 5   | ✔      | *Completed tasks from 8-9 days*<br>Research: FAISS docs, tutorials<br>Fixed action_save()<br><br>FAISS Integration:<br>- replace manual search with FAISS, top-1 match<br><br>Tracking:<br>- simple tracking: assign ID per face, match by bbox proximity<br>- reuse identity across frames Goal: labels stop jumping                                                                                              | 3.5h |
| 6   | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 7   | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 8   | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 9   | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 10  | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 11  | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 12  | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 13  | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |
| 14  | ➖      |                                                                                                                                                                                                                                                                                                                                                                                                                    |      |

  


---

General Path
1. ~~OpenCV webcam feed~~
2. ~~InsightFace detection + embeddings~~
3. ~~Store embeddings locally~~
4. ~~FAISS search~~
5. Add threshold logic
6. Add tracking (basic)
7. Add FastAPI
8. Add logging
9. Optimize (ONNX)
10. Add SQL
11. Dockerize