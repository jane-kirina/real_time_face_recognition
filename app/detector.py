import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import insightface
from insightface.app import FaceAnalysis

# Detect Faces
def init_face_detector(model_name='buffalo_s', 
                       allowed_modules=['detection', 'recognition'], 
                       det_size=(256, 256)):
    # Initialize InsightFace face detector

    app = FaceAnalysis(model_name, 
                       allowed_modules=allowed_modules)
    
    # 0 = GPU, -1 = CPU
    # default det_size=(640, 640))
    app.prepare(ctx_id=-1, det_size=det_size)

    return app

def detect_faces(detector, frame):
    # Detect faces in the given frame
    return detector.get(frame)