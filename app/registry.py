from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np

'''
FaceRegistry is an in-memory database of faces + search for them
Management center for all faces in the system that responsible for:
- storage of people (labels)
- storage of embeddings
- quick search via FAISS (index)
- database update (add/delete)
- synchronization with file (.npy)
'''
class FaceRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        # thread safety: the API can simultaneously add person, search and delete
        # and ithout lock you can break data
        self.lock = RLock()

        self.labels = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = None
        self.version = 0

    def load(self):
        # Loads the database from disk into memory
        with self.lock:
            # If db doesn't exist -> create new
            if not self.db_path.exists():
                self.labels = []
                self.embeddings = np.empty((0, 512), dtype=np.float32)
                self.index = None
                self.embedding_dim = 512
                return

            data = np.load(self.db_path, allow_pickle=True).item()

            self.labels = list(data.get('labels', []))
            self.embeddings = np.asarray(data.get('embeddings', []), dtype=np.float32)

            # no embeddings -> no index
            if self.embeddings.size == 0:
                self.embeddings = np.empty((0, 512), dtype=np.float32)
                self.index = None
                self.embedding_dim = 512
            # esle: normize data(makes cosine similarity correct) and built index
            else:
                self.embeddings = self.normalize(self.embeddings)
                self.embedding_dim = self.embeddings.shape[1]
                self.index = self.build_index(self.embeddings)

            self.version += 1

    def reload(self):
        self.load()

    def add_person(self, name, embedding: np.ndarray):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        embedding = self.normalize(embedding)

        with self.lock:
            if self.embeddings is None or self.embeddings.size == 0:
                self.embeddings = embedding
                self.labels = [name]
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
                self.labels.append(name)

            self.save()
            self.index = self.build_index(self.embeddings)
            self.embedding_dim = self.embeddings.shape[1]
            self.version += 1

    def delete_person(self, name):
        with self.lock:
            indices = [i for i, l in enumerate(self.labels) if l != name]
            deleted = len(self.labels) - len(indices)

            if deleted == 0:
                return 0

            self.labels = [self.labels[i] for i in indices]

            if indices:
                self.embeddings = self.embeddings[indices]
                self.embeddings = self.normalize(self.embeddings)
                self.index = self.build_index(self.embeddings)
            else:
                self.embeddings = np.empty((0, 512), dtype=np.float32)
                self.index = None

            self.save()
            self.version += 1
            return deleted

    def search(self, embedding: np.ndarray, top_k: int = 1):
        with self.lock:
            if self.index is None or not self.labels:
                return []

            query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
            query = self.normalize(query)

            scores, indices = self.index.search(query, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.labels):
                    continue
                results.append({
                    'label': self.labels[idx],
                    'score': float(score)
                })

            return results

    def stats(self) -> dict[str, Any]:
        with self.lock:
            return {
                'persons': len(self.labels),
                'embedding_dim': self.embedding_dim,
                'index_ready': self.index is not None,
                'version': self.version,
                'db_path': str(self.db_path)
            }

    def build_index(self, embeddings: np.ndarray):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def normalize(self, embeddings: np.ndarray):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norm, 1e-12, None)

    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self.db_path.with_suffix('.tmp.npy')
        np.save(tmp, {
            'labels': self.labels,
            'embeddings': self.embeddings
        }, allow_pickle=True)

        tmp.replace(self.db_path)

def get_snapshot(self):
    with self.lock:
        return {
            'labels': list(self.labels),
            'index': self.index,
            'version': self.version,
            'embedding_dim': self.embedding_dim,
        }