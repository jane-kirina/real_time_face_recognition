import numpy as np

from app.embedding import (
    l2_normalize,
    build_faiss_index,
    find_best_match_faiss,
)

def test_l2_normalize_returns_unit_vector():
    vec = np.array([3.0, 4.0], dtype=np.float32)

    result = l2_normalize(vec)

    norm = np.linalg.norm(result)

    assert result.shape == vec.shape
    assert np.isclose(norm, 1.0)
    assert np.allclose(result, np.array([0.6, 0.8], dtype=np.float32))

def test_l2_normalize_handles_zero_vector():
    vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    result = l2_normalize(vec)

    # should not crash, and usually stays zero
    assert np.all(result == 0.0)

def test_build_faiss_index_empty():
    embeddings = []
    names = []

    index, out_names = build_faiss_index(embeddings, names)

    assert index is None or index.ntotal == 0
    assert out_names == []

def test_build_faiss_index_single_vector():
    embeddings = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    names = ['alice']

    index, out_names = build_faiss_index(embeddings, names)

    assert index is not None
    assert index.ntotal == 1
    assert out_names == ['alice']

def test_find_best_match_returns_correct_name():
    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]
    names = ['alice', 'bob']

    index, names = build_faiss_index(embeddings, names)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score > 0.9

def test_find_best_match_handles_empty_index():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, None, [])

    assert best_name == 'unknown'
    assert best_score == 0.0

def test_find_best_match_prefers_closest_vector():
    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32), # alice
        np.array([0.0, 1.0, 0.0], dtype=np.float32), # bob
    ]
    names = ['alice', 'bob']

    index, names = build_faiss_index(embeddings, names)

    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score > 0.8

def test_find_best_match_returns_low_score_for_different_vector():
    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ]
    names = ['alice']

    index, names = build_faiss_index(embeddings, names)

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    best_name, best_score = find_best_match_faiss(query, index, names)

    assert best_name == 'alice'
    assert best_score < 0.5