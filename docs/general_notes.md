# Notes(theory)

## Face Recognition 
Facial recognition is the process of converting a facial image into a numerical vector and comparing it to a database to identify or verify a person

- Camera -> face -> numbers -> comparison -> result
- Face -> vector (embedding) -> distance -> match / no match
- The neural network (CNN) does it all
- System logic:
  - Find face -> align -> encode -> compare

General pipeline:
1. Detect face
2. Align / crop
3. Extract embedding (vector)
4. Compare with database (cosine similarity)
5. Apply threshold → known / unknown

### Clustering of faces helps simplify and speed up identification
1. Clustering with Visual Attributes
- Features (gender, age, glasses, etc.) are used -> these features help narrow the search space -> hierarchical (agglomerative) clustering is used
- Result: more features -> better the clustering quality
2. Clustering for Identification (1:N)
- Used in open-set recognition (the person may be unknown)
- Works with video (multiple frames of the same face)
- Clusters combine different images of the same person
- Result: Increasing the number of clusters initially improves the result, but after a certain point, it worsens the result

Peak performance - the number of clusters:
- too few -> different people in one cluster
- too many -> one person is split into several

## Face embeddings
> Embeddings = numerical representation of faces

Face recognition models embedding spaces are strongly structured by semantic facial attributes. Age, hair, glasses, etc. are attributes actively shape distances and geometry in the embedding space, not just identity labels.

1. Embeddings are structured, not random. Points cluster by identity but also align along attribute axes
2. Attributes create 'directions'. Changing one attribute (for exmple adding glasses) -> moves embeddings in a consistent direction
3. Bias & fairness implications. If attributes affect distances ->m odels may unintentionally encode bias. Example: glasses or hairstyle altering similarity scores
4. Trade-off: invariance vs sensitivity: Good FR systems ignore irrelevant variation, but some attributes still leak into embeddings. (robustness vs interpretability)

## 0 - Resources used
This section is primarily intended to save everything that helped in creating the project, for future reference and to refresh own knowledge

facial recognition:
- https://www.researchgate.net/publication/316538740_Face_Identification_and_Clustering
- https://www.researchgate.net/publication/220634629_Face_Matching_and_Retrieval_in_Forensics_Applications


Face Embedding:
- [Attributes Shape the Embedding Space of Face Recognition Models](https://arxiv.org/html/2507.11372v1)
- [Face Embedding and what you need to know](https://uysim.medium.com/face-embedding-and-what-you-need-to-know-a623c7111b5)
- [facial recognition & synthetic dataset](https://www.sciencedirect.com/science/article/pii/S095219762400099X)

Tech:
- ArcFace outperforms FaceNet: [Face Recognition Using ArcFace and FaceNet in Google
Cloud Platform For Attendance System Mobile
Application](https://www.researchgate.net/publication/368485018_Face_Recognition_Using_ArcFace_and_FaceNet_in_Google_Cloud_Platform_For_Attendance_System_Mobile_Application)

Articles:
- [Facial Analysis with “insightface” library](https://medium.com/@appanamukesh77/comprehensive-insights-onfacial-analysis-with-insightface-library-796d80464f45)
- [ArcFace: Facial Recognition Model](https://medium.com/analytics-vidhya/arcface-facial-recognition-model-2eb77080aa80)

Docs:
- https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
- [InsightFace](https://github.com/deepinsight/insightface/tree/master)
- [Haar Cascades](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
