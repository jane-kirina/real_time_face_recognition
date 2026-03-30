import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import insightface
from insightface.app import FaceAnalysis

# Detect Faces
def init_face_detector(model_name='buffalo_s', 
                       allowed_modules=['detection', 'recognition'], 
                       ctx_id=-1, # 0 = GPU, -1 = CPU
                       det_size=(256, 256)): # default det_size=(640, 640))
    # Initialize InsightFace face detector

    app = FaceAnalysis(model_name, 
                       allowed_modules=allowed_modules)
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    return app

def detect_faces(detector, frame):
    # Detect faces in the given frame
    return detector.get(frame)