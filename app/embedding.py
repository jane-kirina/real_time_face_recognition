import os
import numpy as np
import faiss

def l2_normalize(vec):
    # Normalize vector to unit length

    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)

    return vec / norm if norm != 0 else vec

def cosine_similarity(a, b):
    # Calculate cosine similarity for normalized vectors
    return np.dot(a, b)

# ----------------------------
# save, load & delete embedding

def save_embedding(db, person_name, embedding):
    # Save one embedding for a person in the database
    embedding = l2_normalize(embedding)

    if person_name not in db:
        db[person_name] = []

    db[person_name].append(embedding)

def save_db(db, path='data/face_db.npy'):
    # Save embeddings database to file

    np.save(path, db, allow_pickle=True)

def load_db(path='data/face_db.npy'):
    # Load embeddings database from file

    if not os.path.exists(path):
        return {}
    return np.load(path, allow_pickle=True).item()

def delete_person(db, name):
    if name in db:
        del db[name]
 
def list_persons(db):
    return sorted(db.keys())


# ----------------------------
# find best match

def find_best_match(query_embedding, db):
    # Find the most similar person in the databas
    
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

# ----------------------------
# FAISS search

def flatten_db(db):
    # Convert dict database to flat embeddings array and names list
    # db -> embeddings + names

    all_embeddings = []
    names = []

    for person_name, embeddings in db.items():
        for emb in embeddings:
            emb = l2_normalize(emb)
            all_embeddings.append(emb)
            names.append(person_name)

    if not all_embeddings:
        return np.empty((0, 512), dtype=np.float32), []

    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
    return all_embeddings, names

def build_faiss_index(db):
    # Build FAISS index from dict database
    
    known_embeddings, names = flatten_db(db)

    if len(names) == 0:
        return None, names

    dim = known_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(known_embeddings)

    return index, names

def find_best_match_faiss(query_embedding, index, names):
    # Find best match using FAISS, top-1 match

    if index is None or len(names) == 0:
        return None, -1.0

    query_embedding = l2_normalize(query_embedding)
    query_embedding = np.asarray([query_embedding], dtype=np.float32)

    scores, indices = index.search(query_embedding, k=1)

    best_score = float(scores[0][0])
    best_index = int(indices[0][0])

    if best_index < 0:
        return None, -1.0

    best_name = names[best_index]
    return best_name, best_score