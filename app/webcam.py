import cv2

# Custom
from app.detector import (detect_faces)
from app.drawer import (draw_fps, draw_faces, draw_paused)
from app.embedding import (load_db, build_faiss_index, find_best_match_faiss)
from app.handler_keyboard import handle_keypress_action
from app.tracker import (update_track_identity, update_tracks)

# ----------------------------
# Frame processing
# ----------------------------

def read_frame(stream):
    # Read one frame from webcam stream

    ret, frame = stream.read()
    if not ret:
        print('Failed to read frame from webcam')
        return None
    return frame

# ----------------------------
# Responsible for the content of the frame: fps, paused text, embedding
# ----------------------------
def build_display_frame(state):
    # Prepare frame for display with boxes, labels and FPS
    # Main logic for adding text, faces detection on frame

    if state['frame'] is None:
        return None

    display_frame = state['frame'].copy()

    draw_faces(display_frame, state['faces'], state['scale'])
    draw_fps(display_frame, state.get('fps', 0))

    if state.get('paused', False):
        draw_paused(display_frame)

    return display_frame

def find_track_by_id(tracks, track_id):
    for track in tracks:
        if track['id'] == track_id:
            return track
    return None

def process_embeddings(state, match_threshold=0.5):
    # Match detected face embeddings with saved people
    for face in state['faces']:
        track = find_track_by_id(state['tracks'], getattr(face, 'track_id', None))
        if track is None:
            continue

        if getattr(face, 'embedding', None) is None:
            face.name = track['name']
            face.match_score = track['score']
            continue

        best_name, best_score = find_best_match_faiss(
            face.embedding,
            state['faiss_index'],
            state['faiss_names']
        )

        if best_score >= match_threshold:
            predicted_name = best_name
        else:
            predicted_name = 'unknown'

        update_track_identity(track, predicted_name, best_score, match_threshold)

        face.name = track['name']
        face.match_score = track['score']

# ----------------------------
# Main function:
# - Webcam stream, video loop with fps, embeddings
# ----------------------------
# Webcam stream, video loop with fps
def start_camera(fps_counter, face_detector, scale = 0.5, detect_every_n_frames = 3, match_threshold = 0.5):
    # Start webcam stream and process frames in a loop

    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        raise RuntimeError('No stream: Could not open webcam')
    
    window_name = 'Webcam - Live'
    db = load_db()
    faiss_index, faiss_names = build_faiss_index(db)

    state = { # Keep information about one frame: frame itself, frame_id, FPS, etc.
        'paused': False,
        'frame': None,
        'fps': 0,
        'scale': scale,
        'frame_id': 0,
        'faces': [],
        'display_frame': None,
        'db': db,
        'faiss_index': faiss_index,
        'faiss_names': faiss_names,
        'tracks':[],
        'next_track_id': 1
    }

    while True:
        if not state['paused']: # TODO separeted func
            frame = read_frame(stream)

            if frame is None:
                break
            
            # ----------------------------
            # Frame processing

            # Optimizations: Create small frame for detection 
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Optimizations: Detection will be every N frames
            if state['frame_id'] % detect_every_n_frames == 0:
                state['faces'] = detect_faces(face_detector, small_frame)

                state['tracks'], state['next_track_id'] = update_tracks(
                    state['tracks'],
                    state['faces'],
                    state['next_track_id'],
                    max_distance=40,
                    max_missed=8
                    )

            process_embeddings(state, match_threshold)
            
            # Update state with frame, fps
            state['frame'] = frame.copy()
            state['fps'] = fps_counter.update()
            state['frame_id'] += 1
        # ----------------------------
        # Draw information on a frame
        display_frame = build_display_frame(state)

        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        state['display_frame'] = display_frame

        # ----------------------------
        # Keypress handle
        result = handle_keypress_action(state)
        if result == 'exit':
            break

    stream.release()
    cv2.destroyAllWindows()
