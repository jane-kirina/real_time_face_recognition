# Notes(theory)

## 1 - Face Recognition 
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

Peak performance: the number of clusters:
- too few -> different people in one cluster
- too many -> one person is split into several


## 0 - Resources used
This section is primarily intended to save everything that helped in creating the project, for future reference and to refresh own knowledge

facial recognition:
- https://www.researchgate.net/publication/316538740_Face_Identification_and_Clustering
- https://www.researchgate.net/publication/220634629_Face_Matching_and_Retrieval_in_Forensics_Applications
- 

Docs:
- https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
- [InsightFace](https://github.com/deepinsight/insightface/tree/master)
