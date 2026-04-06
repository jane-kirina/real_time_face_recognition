import numpy as np

from app.embedding import (
    l2_normalize,
    flatten_db,
    build_faiss_index,
    find_best_match_faiss
)

def test_l2_normalize_returns_unit_vector():
    vec = np.array([3.0, 4.0], dtype=np.float32)

    result = l2_normalize(vec)

    assert result.shape == vec.shape
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert np.allclose(result, np.array([0.6, 0.8], dtype=np.float32))

def test_l2_normalize_handles_zero_vector():
    vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    result = l2_normalize(vec)

    assert np.array_equal(result, vec)
    assert result.dtype == np.float32

def test_flatten_db_empty_returns_empty_embeddings_and_names():
    embeddings, names = flatten_db({})

    assert embeddings.shape == (0, 512)
    assert embeddings.dtype == np.float32
    assert names == []

def test_flatten_db_returns_normalized_embeddings_and_names():
    db = {
        'alice': [np.array([3.0, 4.0, 0.0], dtype=np.float32)],
        'bob': [np.array([0.0, 5.0, 0.0], dtype=np.float32)]
    }

    embeddings, names = flatten_db(db)

    assert embeddings.shape == (2, 3)
    assert names == ['alice', 'bob']
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0)
    assert np.isclose(np.linalg.norm(embeddings[1]), 1.0)
    assert np.allclose(embeddings[0], np.array([0.6, 0.8, 0.0], dtype=np.float32))

def test_build_faiss_index_empty_db():
    index, names = build_faiss_index({})

    assert index is None
    assert names == []

def test_build_faiss_index_single_embedding():
    db = {
        'alice': [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    }

    index, names = build_faiss_index(db)

    assert index is not None
    assert index.ntotal == 1
    assert names == ['alice']

def test_build_faiss_index_multiple_embeddings_keeps_all_names():
    db = {
        'alice': [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.0], dtype=np.float32)
        ],
        'bob': [
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ]
    }

    index, names = build_faiss_index(db)

    assert index is not None
    assert index.ntotal == 3
    assert names == ['alice', 'alice', 'bob']

def test_find_best_match_faiss_returns_best_name_for_exact_match():
    db = {
        'alice': [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        'bob': [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    }
    index, names = build_faiss_index(db)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score > 0.99

def test_find_best_match_faiss_returns_none_and_minus_one_for_empty_index():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, None, [])

    assert best_name is None
    assert best_score == -1.0

def test_find_best_match_faiss_prefers_closest_vector():
    db = {
        'alice': [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        'bob': [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    }
    index, names = build_faiss_index(db)

    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score > 0.9

def test_find_best_match_faiss_returns_low_score_for_different_vector():
    db = {
        'alice': [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    }
    index, names = build_faiss_index(db)

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score < 0.1
    