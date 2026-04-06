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
> 
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

## Configuration

The project uses environment-based configuration via `.env`.

1. Copy `.env.example` to `.env`
2. Adjust thresholds, paths, and runtime settings
3. Run the application

Example settings:
- `MODEL_NAME`
- `MATCH_THRESHOLD`
- `DETECT_EVERY_N_FRAMES`
- `FACE_DB_PATH`
- `API_PORT`

