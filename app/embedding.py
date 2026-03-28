import os
import numpy as np

# embedding normalization
def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)

    return vec / norm if norm != 0 else vec

# comparison by cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b)

# ----------------------------
# save & load embedding

def save_embedding(db, person_name, embedding):
    embedding = l2_normalize(embedding)

    if person_name not in db:
        db[person_name] = []

    db[person_name].append(embedding)

def save_db(db, path='face_db.npy'):
    np.save(path, db, allow_pickle=True)

def load_db(path='face_db.npy'):
    if not os.path.exists(path):
        return {}
    return np.load(path, allow_pickle=True).item()

# ----------------------------
# find best match

def find_best_match(query_embedding, db):
    query_embedding = l2_normalize(query_embedding)

    best_name = None
    best_score = -1.0

    for person_name, embeddings in db.items():
        for emb in embeddings:
            score = cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_name = person_name

    return best_name, best_score