import numpy as np
from app.tracker import (
    add_prediction_to_track,
    center_distance,
    get_smoothed_identity,
    update_track_identity,
    update_tracks,
)


class DummyFace:
    def __init__(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.track_id = None


def test_center_distance_returns_zero_for_same_bbox():
    bbox = [10, 10, 50, 50]

    result = center_distance(bbox, bbox)

    assert result == 0.0


def test_center_distance_uses_bbox_centers():
    bbox1 = [0, 0, 10, 10]
    bbox2 = [10, 0, 20, 10]

    result = center_distance(bbox1, bbox2)

    assert result == 10.0


def test_update_tracks_reuses_existing_track():
    tracks = [
        {
            'id': 1,
            'bbox': [10, 10, 50, 50],
            'name': 'unknown',
            'score': 0.0,
            'missed': 0,
        }
    ]
    faces = [DummyFace([12, 12, 52, 52])]

    updated_tracks, next_track_id = update_tracks(
        tracks=tracks,
        faces=faces,
        next_track_id=2,
        max_distance=20,
        max_missed=3,
    )

    assert len(updated_tracks) == 1
    assert updated_tracks[0]['id'] == 1
    assert updated_tracks[0]['bbox'] == [12, 12, 52, 52]
    assert updated_tracks[0]['missed'] == 0
    assert faces[0].track_id == 1
    assert next_track_id == 2


def test_update_tracks_creates_new_track_for_far_face():
    tracks = []
    faces = [DummyFace([100, 100, 160, 160])]

    updated_tracks, next_track_id = update_tracks(
        tracks=tracks,
        faces=faces,
        next_track_id=1,
        max_distance=20,
        max_missed=3,
        smoothing_window=5,
    )

    assert len(updated_tracks) == 1
    assert updated_tracks[0]['id'] == 1
    assert updated_tracks[0]['bbox'] == [100, 100, 160, 160]
    assert updated_tracks[0]['name'] == 'unknown'
    assert updated_tracks[0]['score'] == 0.0
    assert updated_tracks[0]['missed'] == 0
    assert 'history' in updated_tracks[0]
    assert updated_tracks[0]['history'].maxlen == 5
    assert faces[0].track_id == 1
    assert next_track_id == 2


def test_update_tracks_increments_missed_for_unmatched_track():
    tracks = [
        {
            'id': 1,
            'bbox': [10, 10, 50, 50],
            'name': 'unknown',
            'score': 0.0,
            'missed': 0,
        }
    ]
    faces = []

    updated_tracks, next_track_id = update_tracks(
        tracks=tracks,
        faces=faces,
        next_track_id=2,
        max_distance=20,
        max_missed=3,
    )

    assert len(updated_tracks) == 1
    assert updated_tracks[0]['id'] == 1
    assert updated_tracks[0]['missed'] == 1
    assert next_track_id == 2


def test_update_tracks_removes_stale_tracks():
    tracks = [
        {
            'id': 1,
            'bbox': [10, 10, 50, 50],
            'name': 'unknown',
            'score': 0.0,
            'missed': 3,
        }
    ]
    faces = []

    updated_tracks, next_track_id = update_tracks(
        tracks=tracks,
        faces=faces,
        next_track_id=2,
        max_distance=20,
        max_missed=3,
    )

    assert updated_tracks == []
    assert next_track_id == 2


def test_update_track_identity_sets_name_for_unknown_track():
    track = {
        'id': 1,
        'bbox': [10, 10, 50, 50],
        'name': 'unknown',
        'score': 0.0,
        'missed': 0,
    }

    update_track_identity(track, 'alice', 0.82, match_threshold=0.5)

    assert track['name'] == 'alice'
    assert track['score'] == 0.82


def test_update_track_identity_ignores_weak_prediction():
    track = {
        'id': 1,
        'bbox': [10, 10, 50, 50],
        'name': 'alice',
        'score': 0.80,
        'missed': 0,
    }

    update_track_identity(track, 'bob', 0.40, match_threshold=0.5)

    assert track['name'] == 'alice'
    assert track['score'] == 0.80


def test_update_track_identity_updates_score_for_same_name_if_better():
    track = {
        'id': 1,
        'bbox': [10, 10, 50, 50],
        'name': 'alice',
        'score': 0.75,
        'missed': 0,
    }

    update_track_identity(track, 'alice', 0.83, match_threshold=0.5)

    assert track['name'] == 'alice'
    assert track['score'] == 0.83


def test_update_track_identity_does_not_switch_on_small_score_gain():
    track = {
        'id': 1,
        'bbox': [10, 10, 50, 50],
        'name': 'alice',
        'score': 0.80,
        'missed': 0,
    }

    update_track_identity(track, 'bob', 0.84, match_threshold=0.5)

    assert track['name'] == 'alice'
    assert track['score'] == 0.80


def test_update_track_identity_switches_on_clear_score_gain():
    track = {
        'id': 1,
        'bbox': [10, 10, 50, 50],
        'name': 'alice',
        'score': 0.80,
        'missed': 0,
    }

    update_track_identity(track, 'bob', 0.90, match_threshold=0.5)

    assert track['name'] == 'bob'
    assert track['score'] == 0.90


def test_add_prediction_to_track_appends_history():
    track = {'history': []}

    add_prediction_to_track(track, 'alice', 0.91)

    assert len(track['history']) == 1
    assert track['history'][0]['name'] == 'alice'
    assert track['history'][0]['score'] == 0.91


def test_get_smoothed_identity_returns_unknown_for_empty_history():
    track = {'history': []}

    name, score = get_smoothed_identity(track, match_threshold=0.5, min_votes=2)

    assert name == 'unknown'
    assert score == 0.0


def test_get_smoothed_identity_returns_unknown_when_not_enough_votes():
    track = {'history': []}
    add_prediction_to_track(track, 'alice', 0.80)
    add_prediction_to_track(track, 'bob', 0.82)

    name, score = get_smoothed_identity(track, match_threshold=0.5, min_votes=2)

    assert name == 'unknown'
    assert score == 0.0


def test_get_smoothed_identity_uses_majority_vote_and_average_score():
    track = {'history': []}
    add_prediction_to_track(track, 'alice', 0.80)
    add_prediction_to_track(track, 'alice', 0.90)
    add_prediction_to_track(track, 'bob', 0.95)
    add_prediction_to_track(track, 'unknown', 0.99)
    add_prediction_to_track(track, 'alice', 0.70)

    name, score = get_smoothed_identity(track, match_threshold=0.5, min_votes=2)

    assert name == 'alice'
    assert np.isclose(score, (0.80 + 0.90 + 0.70) / 3)


def test_get_smoothed_identity_ignores_predictions_below_threshold():
    track = {'history': []}
    add_prediction_to_track(track, 'alice', 0.40)
    add_prediction_to_track(track, 'alice', 0.45)
    add_prediction_to_track(track, 'bob', 0.49)

    name, score = get_smoothed_identity(track, match_threshold=0.5, min_votes=2)

    assert name == 'unknown'
    assert score == 0.0