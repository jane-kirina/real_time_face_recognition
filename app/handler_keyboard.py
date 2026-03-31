import cv2
import os

# Custom
from app.embedding import (save_db, load_db, save_embedding, build_faiss_index)

# ----------------------------
# Keypress handle:
# - pause the frame
# - save the frame
# - exit the program
# ----------------------------
def action_exit(state):
    # Stop the webcam loop

    return 'exit'

def action_pause(state):
    # Switch pause mode on or off

    state.paused = not state.paused
    print(f"Paused: {state.paused}")

def action_save(state):
    # Save the current frame to the outputs folder

    if state.frame is None:
        print('No frame to save')
        return

    # create folder for images
    os.makedirs('outputs', exist_ok=True)

    filename = f"outputs/frame_{state.frame_id}.jpg"
    success = cv2.imwrite(filename, state.display_frame)
    
    # Print if the frame was saved and the name of frame
    state.logger.log_system('SAVED_FRAME', success=success, filename=filename)
    print(f"Saved: {success}, {filename}")

def action_save_embedding(state):
    # Save embedding for one detected face

    faces = state.faces
    db = state.db
    
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
    
    person_name = input('Enter person name: ').strip()

    if not person_name:
        print('Empty name. Enrollment cancelled')
        return
    
    if person_name in db and len(db[person_name]) >= 10:
        state.logger.log_system('MAXIMUM_EMBEDDINGS', name=person_name)
        print(f'{person_name} has the maximum 8 embeddings')
        return
    
    save_embedding(db, person_name, face.embedding)
    save_db(db)
    state.logger.log_system('SAVED_NEW_EMBEDDING', name=person_name)

def action_reload_database(state):
    try:
        state.db = load_db()
        state.faiss_index, state.faiss_names = build_faiss_index(state.db)

        total_vectors = (
            state.faiss_index.ntotal
            if state.faiss_index is not None
            else 0
        )

        state.logger.log_system(
            'DATABASE_RELOADED',
            total_persons=len(state.db),
            total_vectors=total_vectors,
        )

        print('Database reloaded')

    except Exception as e:
        print(f'Reload failed: {e}')

KEY_ACTIONS = {
    ord('q'): action_exit,
    27: action_exit, # ESC key
    ord('s'): action_save,
    ord('p'): action_pause,
    ord('e'): action_save_embedding,
    ord('r'): action_reload_database
}

def handle_keypress_action(state):
    # Handle keyboard input and run mapped action

    key = cv2.waitKey(1) & 0xFF
    action = KEY_ACTIONS.get(key & 0xFF)

    if action is None:
        return None

    return action(state)
