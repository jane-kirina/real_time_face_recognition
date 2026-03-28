import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

print("start") # TODO for debug
import insightface
print("insightface ok") # TODO for debug
from insightface.app import FaceAnalysis
print("FaceAnalysis ok") # TODO for debug

print('insightface', insightface.__version__)

# Detect Faces
def init_face_detector(model_name='buffalo_s', 
                       allowed_modules=['detection', 'recognition'], 
                       det_size=(256, 256)):
    app = FaceAnalysis(model_name, 
                       allowed_modules=allowed_modules)
    
    # 0 = GPU, -1 = CPU
    # default det_size=(640, 640))
    app.prepare(ctx_id=-1, det_size=det_size)

    return app

def detect_faces(detector, frame):
    return detector.get(frame)


def extract_face_embedding(face):
    return face.embedding