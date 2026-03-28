import cv2

# Custom
from app.detector import detect_faces

# Util funcs

def draw_fps(frame, fps):
    cv2.putText(
        frame, # img
        f"FPS: {int(fps)}", # text
        (10, 30), # org
        cv2.FONT_HERSHEY_SIMPLEX, # fontFace
        1, # fontScale
        (255, 0, 0), # color, BGR
        2 # thickness
    )

def draw_faces(frame, faces, scale):
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Webcam stream, video loop with fps
def start_camera(fps_counter, face_detector):
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        raise RuntimeError('No stream: Could not open webcam')
    
    window_name = 'Webcam - Live'
    scale = 0.5

    while True:
        ret, frame = stream.read()

        if not ret:
            print('Failed to read frame from webcam')
            # raise RuntimeError('Failed to read frame from webcam') # TODO
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        faces = detect_faces(face_detector, small_frame)
        draw_faces(frame, faces, scale)
        
        fps = fps_counter.update()
        draw_fps(frame, fps)
        
        cv2.imshow(window_name, frame)

        # to exit stream
        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('й'):
            break

    stream.release()
    cv2.destroyAllWindows()
