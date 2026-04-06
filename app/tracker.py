import math
from collections import Counter, deque

def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def center_distance(bbox1, bbox2):
    c1x, c1y = get_bbox_center(bbox1)
    c2x, c2y = get_bbox_center(bbox2)
    return math.hypot(c1x - c2x, c1y - c2y)

def update_tracks(tracks, faces, next_track_id, max_distance=50, max_missed=10, smoothing_window=5):
    # keep track of which track IDs have already been matched on this frame
    matched_track_ids = set()

    # go through each detected face on the current frame
    for face in faces:
        # convert bbox to simple int list: [x1, y1, x2, y2]
        face_bbox = face.bbox.astype(int).tolist()

        # try to find the closest existing track
        best_track = None
        best_distance = float('inf') # start with very large distance

        # compare current face with all existing tracks
        for track in tracks:
            # skip tracks that were already assigned to another face
            if track['id'] in matched_track_ids:
                continue

            # distance between current face and track (by bbox centers)
            dist = center_distance(face_bbox, track['bbox'])
            # pick the closest track within allowed distance
            if dist < best_distance and dist < max_distance:
                best_distance = dist
                best_track = track
        
        # found a matching track -> reuse it
        if best_track is not None:
            # update track position with current bbox
            best_track['bbox'] = face_bbox
            # reset "missed" counter since we see this track again
            best_track['missed'] = 0
            # assign existing track ID to this face
            face.track_id = best_track['id']
            # mark this track as already used
            matched_track_ids.add(best_track['id'])
        else:
            # no matching track -> create a new one
            new_track = {
                'id': next_track_id,
                'bbox': face_bbox,
                'name': 'unknown',  # will be updated later by recognition
                'score': 0.0,
                'missed': 0,
                'history': deque(maxlen=smoothing_window)  # later replace 5 with config value
            }
            
            # add new track to list
            tracks.append(new_track)
            # assign new ID to face
            face.track_id = next_track_id
            # mark this new track as used
            matched_track_ids.add(next_track_id)
            # prepare ID for next new track
            next_track_id += 1

    # increase 'missed' counter for tracks that were not seen on this frame
    for track in tracks:
        if track['id'] not in matched_track_ids:
            track['missed'] += 1

    # remove tracks that disappeared for too long
    tracks = [track for track in tracks if track['missed'] <= max_missed]

    # return updated tracks and next available ID
    return tracks, next_track_id

def update_track_identity(track, predicted_name, predicted_score, match_threshold=0.5):
    # if prediction is weak or unknown -> ignore it (keep previous identity)
    if predicted_name == 'unknown' or predicted_score < match_threshold:
        return

    # if track does not have a name yet -> assign it immediately
    if track['name'] == 'unknown':
        track['name'] = predicted_name
        track['score'] = predicted_score
        return

    # if predicted name matches current track name -> update score if better
    if predicted_name == track['name']:
        if predicted_score > track['score']:
            track['score'] = predicted_score
        return

    # if model suggests a different person:
    # only switch identity if confidence is significantly higher
    if predicted_score > track['score'] + 0.05:
        track['name'] = predicted_name
        track['score'] = predicted_score

def add_prediction_to_track(track, predicted_name, predicted_score):
    track['history'].append({
        'name': predicted_name,
        'score': float(predicted_score),
    })

def get_smoothed_identity(track, match_threshold=0.5, min_votes=2):
    history = track.get('history', [])
    if not history:
        return 'unknown', 0.0

    # ignore weak predictions
    valid = [item for item in history if item['score'] >= match_threshold]
    if not valid:
        return 'unknown', 0.0

    # majority vote by name
    names = [item['name'] for item in valid if item['name'] != 'unknown']
    if not names:
        return 'unknown', 0.0

    counts = Counter(names)
    best_name, votes = counts.most_common(1)[0]

    if votes < min_votes:
        return 'unknown', 0.0

    # average confidence only for the winning name
    winning_scores = [item['score'] for item in valid if item['name'] == best_name]
    avg_score = sum(winning_scores) / len(winning_scores)

    return best_name, avg_score