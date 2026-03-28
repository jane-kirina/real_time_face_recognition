import cv2

# Custom
from app.detector import (detect_faces)
from app.drawer import (draw_fps, draw_faces, draw_paused)
from app.embedding import (load_db, find_best_match)
from app.handler_keyboard import handle_keypress_action

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

    draw_faces(
        display_frame,
        state.get('faces', []),
        state.get('scale', 1.0)
    )
    draw_fps(display_frame, state.get('fps', 0))

    if state.get('paused', False):
        draw_paused(display_frame)

    return display_frame


def process_embeddings(state, match_threshold=0.5):
    # Match detected face embeddings with saved people

    for face in state['faces']:
        if getattr(face, 'embedding', None) is None:
            face.name = 'no embedding'
            face.match_score = 0.0
            continue

        best_name, best_score = find_best_match(face.embedding, state['db'])

        if best_score >= match_threshold:
            face.name = best_name
            face.match_score = best_score
        else:
            face.name = 'unknown'
            face.match_score = best_score


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

    state = { # Keep information about one frame: frame itself, frame_id, FPS and other
        'paused': False,
        'frame': None,
        'fps': 0,
        'frame_id': 0,
        'faces': [],
        'db': load_db(),
        'scale': scale
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

        # ----------------------------
        # Keypress handle
        result = handle_keypress_action(state)
        if result == 'exit':
            break

    stream.release()
    cv2.destroyAllWindows()
