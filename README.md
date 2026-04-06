# Real-Time Face Recognition with Watchlist + Unknown Face Handling

A real-time webcam pipeline that detects faces, tracks them across frames, extracts embeddings, and matches them against a gallery of enrolled identities

For each detected face, the system:
- assigns a stable identity using tracking + temporal smoothing
- shows the predicted name and similarity score
- marks faces as "unknown" if below a similarity threshold
- logs entry/exit events
- stores detections for later inspection via API

Recognition is performed using vector similarity search over stored embeddings

**Key components:**
- real-time face detection and embedding extraction
- track-based identity assignment across frames
- similarity search for identity matching
- threshold-based unknown handling
- temporal smoothing to stabilize predictions
- simple backend for accessing recent events and managing data

**Stack:**
- *InsightFace (ArcFace)* — used for face detection and embedding extraction
- *FAISS* — used to search for the closest identity embedding efficiently
- *OpenCV* — handles frame capture, drawing overlays, and webcam rendering
- *FastAPI* — provides an API for accessing logs and managing enrolled identities

## Architecture
Both parts work through common files with data:
- `data/face_db.npy`
- `data/events.jsonl`

Project has two entrypoints:

1. Camera (online)
2. API (offline)

> The online webcam pipeline and offline API are kept separate because they solve different runtime problems, but both use the same registry abstraction for identity storage and FAISS index management

### Online pipeline
> To run:
```
python main.py
```

1. Read frame from webcam
2. Detect faces
3. Crop / align face
4. Extract embedding
5. Search nearest neighbor in FAISS
6. Apply decision logic (threshold-based classification)
7. Render frame and save event

### Offline pipeline
> To run:
```
uvicorn api:app --host localhost --port 8000 --reload
```
or 
```
python -m app.api
```

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

## Observations
- adding more embeddings per person improves robustness initially, but after several samples the similarity score tends to saturate
- embeddings for the same person form a compact cluster in feature space, so additional samples mostly reinforce the same region rather than improving matching
- beyond a certain point, extra samples reduce variance but do not significantly improve similarity scores
- embeddings are not purely identity-based — attributes like glasses, hairstyle, or lighting can shift vectors and affect similarity scores
- predictions can fluctuate between frames, especially near the threshold -> temporal smoothing was added to stabilize identity
- threshold selection is sensitive to environment conditions (lighting, pose, camera quality)
- tracking improves stability across frames, but can drift if detections are skipped too frequently

### Limitations
- recognition quality depends on the selected model: larger models (e.g. `buffalo_l`) produce more discriminative embeddings, while smaller models trade accuracy for speed
- `buffalo_sc` was chosen to maintain real-time performance on limited hardware, but it results in lower similarity scores and requires more samples per identity for stable matching
- `buffalo_s` offers a better balance between speed and accuracy, but still introduces noticeable performance overhead compared to `buffalo_sc`
- similar-looking faces can produce borderline similarity scores
- threshold is not globally optimal and may require tuning per environment
- tracking may drift if detections are too sparse
- file-based storage is not safe for concurrent writes or large-scale usage

### Problems and trade-offs
- **Real-time performance on limited hardware.** The main challenge was maintaining stable FPS on a non-high-end machine. *Solutions* to balance detection accuracy and runtime speed:
  - added `DETECT_EVERY_N_FRAMES` to reduce detection frequency
  - experimented with FPS control to understand how each component affects performance

- **Model accuracy vs performance.** Larger models improve recognition quality but significantly reduce performance. A smaller model was chosen to keep the system real-time, at the cost of lower similarity scores and reduced robustness

- **Detection cost vs tracking stability.** Running detection on every frame is expensive. *Solution*:
  - skipping frames improves performance, but increases reliance on tracking, which can drift over time

- **Storage simplicity vs scalability.** `.npy` and `.jsonl` were used for simplicity and transparency during development. This makes the system easy to debug, but is not suitable for concurrent or large-scale usage

- **Recognition stability across frames.** Frame-level predictions can fluctuate, especially near the similarity threshold. *Solution*:
  - Temporal smoothing was introduced to stabilize identity assignment across frames.

## Resources used
**Papers**:
- https://www.researchgate.net/publication/316538740_Face_Identification_and_Clustering
- https://www.researchgate.net/publication/220634629_Face_Matching_and_Retrieval_in_Forensics_Applications
- *ArcFace outperforms FaceNet*: [Face Recognition Using ArcFace and FaceNet in Google Cloud Platform For Attendance System Mobile Application](https://www.researchgate.net/publication/368485018_Face_Recognition_Using_ArcFace_and_FaceNet_in_Google_Cloud_Platform_For_Attendance_System_Mobile_Application)

**Articles**:
- [Facial Analysis with “insightface” library](https://medium.com/@appanamukesh77/comprehensive-insights-onfacial-analysis-with-insightface-library-796d80464f45)
- [ArcFace: Facial Recognition Model](https://medium.com/analytics-vidhya/arcface-facial-recognition-model-2eb77080aa80)
- [Centroid Object Tracking](https://medium.com/%40subhajeet.roy/centroid-object-tracking-835d75e4a75a)
- [Object Tracking for Computer Vision](https://datature.io/blog/implementing-object-tracking-for-computer-vision)
- [Tracking Algorithms](https://broutonlab.com/blog/opencv-object-tracking/)
- [Logging](https://realpython.com/python-logging/)
- [Face recognition with OpenCV, Python, and deep learning](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
- [What is a face embedding?](https://www.futurebeeai.com/knowledge-hub/face-embedding-ai)
- [Probabilistic Face Embeddings](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf)
- [Attributes Shape the Embedding Space of Face Recognition Models](https://arxiv.org/html/2507.11372v1)
- [Face Embedding and what you need to know](https://uysim.medium.com/face-embedding-and-what-you-need-to-know-a623c7111b5)
- [facial recognition & synthetic dataset](https://www.sciencedirect.com/science/article/pii/S095219762400099X)

**Docs**:
- [OpenCV](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [InsightFace](https://github.com/deepinsight/insightface/tree/master)
- [Haar Cascades](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
- [FAISS](https://faiss.ai/index.html)
- [Logging](https://docs.python.org/3/library/logging.html)
- [FastAPI](https://fastapi.tiangolo.com/)