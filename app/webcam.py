import cv2
import os

# Custom
from app.detector import (detect_faces)
from app.drawer import (draw_fps, draw_faces, draw_paused)
from app.embedding import load_db, save_db, save_embedding, find_best_match

# ----------------------------
# Keypress handle:
# - pause the frame
# - save the frame
# - exit the program
# ----------------------------
def action_exit(state):
    return 'exit'

def action_pause(state):
    state['paused'] = not state['paused']
    print(f"Paused: {state['paused']}")

def action_save(state):
    if state['frame'] is None:
        print('No frame to save')
        return

    # create folder for images
    os.makedirs('outputs', exist_ok=True)

    filename = f"outputs/frame_{state['frame_id']}.jpg"
    success = cv2.imwrite(filename, state['frame'])
    
    # Print if the frame was saved and the name of frame
    print(f"Saved: {success}, {filename}")

def action_save_embedding(state):
    faces = state.get('faces', [])
    db = state.get('db', {})
    
    if len(faces) == 0:
        print('No face detected')
        return
    
    if len(faces) > 1:
        print('More than one face detected. Only 1 face can be saved')
        return
    
    face = faces[0]
    
    if getattr(face, 'embedding', None) is None:
        print('Face has no embedding')
        return
    
    # TODO hardcoding
    person_name = input('Enter person name: ').strip()

    if not person_name:
        print('Empty name. Enrollment cancelled')
        return
    
    if person_name in db and len(db[person_name]) >= 8:
        print(f'{person_name} has the maximum 8 embeddings')
        return
    
    save_embedding(db, person_name, face.embedding)
    save_db(db)

    print(f'Saved embedding for: {person_name}')

KEY_ACTIONS = { # state => frame
    ord('q'): action_exit,
    27: action_exit, # ESC key
    ord('s'): action_save,
    ord('p'): action_pause,
    ord('e'): action_save_embedding
}

def handle_keypress_action(state):
    key = cv2.waitKey(1) & 0xFF
    action = KEY_ACTIONS.get(key & 0xFF)

    if action is None:
        return None

    return action(state)

# ----------------------------
# Frame processing
# ----------------------------

def read_frame(stream): # TODO ? move back to main loop ?
    ret, frame = stream.read()
    if not ret:
        print('Failed to read frame from webcam')
        # raise RuntimeError('Failed to read frame from webcam') # TODO
        return None
    return frame

# ----------------------------
# Responsible for the content of the frame: fps, paused text, embedding
# ----------------------------
def build_display_frame(state):
    # Main logic for adding text, faces detection on frame
    # TODO move draw_faces & draw_fps here

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


def processing_embeddings(state, MATCH_THRESHOLD=0.5):
    for face in state['faces']:
        if getattr(face, 'embedding', None) is None:
            face.name = 'no embedding'
            face.match_score = 0.0
            continue

        best_name, best_score = find_best_match(face.embedding, state['db'])

        if best_score >= MATCH_THRESHOLD:
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
def start_camera(fps_counter, face_detector, scale = 0.5, DETECT_EVERY_N_FRAMES = 3, MATCH_THRESHOLD = 0.5):
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
        if not state['paused']: # TODO nester code
            frame = read_frame(stream)

            if frame is None:
                break
            
            # ----------------------------
            # Frame processing

            # Optimizations: Create small frame for detection 
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Optimizations: Detection will be every N frames
            if state['frame_id'] % DETECT_EVERY_N_FRAMES == 0:
                state['faces'] = detect_faces(face_detector, small_frame)
            
            processing_embeddings(state, MATCH_THRESHOLD)
            
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
