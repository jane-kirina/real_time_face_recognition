import cv2
import numpy as np
from fastapi import UploadFile

from app.detector import init_face_detector, detect_faces
from app.embedding import load_db, save_db, save_embedding

'''
`offline_pipeline.py` function is to add people to the database  through uploaded images
'''

def decode_upload_to_image(file_bytes):
    '''
    Turns the raw bytes of a file(image) into a regular OpenCV image
    '''
    
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('Failed to decode image')

    return image


def enroll_from_uploads(
    person_name,
    files,
    model_name = 'buffalo_sc',
    det_size = (320, 320),
    max_embeddings_per_person = 10,
) -> dict:
    db = load_db()

    if person_name not in db:
        db[person_name] = []

    if len(db[person_name]) >= max_embeddings_per_person:
        raise ValueError(
            f'{person_name} already has maximum {max_embeddings_per_person} embeddings'
        )

    detector = init_face_detector(model_name=model_name, det_size=det_size)

    saved_count = 0
    skipped = []

    # Loop through all downloaded files
    for upload in files:
        # reads files
        content = upload.file.read()
        # converts to images
        image = decode_upload_to_image(content)
        # detects faces
        faces = detect_faces(detector, image)

        # Validation: there must be exactly one face in the photo
        if len(faces) != 1:
            skipped.append(
                {
                    'filename': upload.filename,
                    'reason': f'expected 1 face, got {len(faces)}'
                }
            )
            continue

        face = faces[0]
        # Validation: checking that the embedding exists
        if getattr(face, 'embedding', None) is None:
            skipped.append(
                {
                    'filename': upload.filename,
                    'reason': 'embedding not found'
                }
            )
            continue

        # Validation: re-checking the limit
        if len(db[person_name]) >= max_embeddings_per_person:
            skipped.append(
                {
                    'filename': upload.filename,
                    'reason': 'person reached max embeddings'
                }
            )
            break

        save_embedding(db, person_name, face.embedding)
        saved_count += 1

    save_db(db)

    # return status
    return {
        'status': 'ok',
        'person_name': person_name,
        'saved_count': saved_count,
        'total_embeddings_for_person': len(db.get(person_name, [])),
        'skipped': skipped,
    }