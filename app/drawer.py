import cv2

# ----------------------------
# Draw on frame smth
# ----------------------------

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
        # scale bbox from small_frame back to original frame
        x1, y1, x2, y2 = (face.bbox / scale).astype(int)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{face.name}: {face.det_score:.2f}"
        if getattr(face, 'embedding', None) is None:
            label = 'no embedding'
        
        cv2.putText(
            frame, # img
            label, # text
            (x1, y1 - 10), # org
            cv2.FONT_HERSHEY_SIMPLEX, # fontFace
            0.6, # fontScale
            (0, 255, 0), # color, BGR
            2 # thickness
        )

def draw_paused(frame):
    text = 'Paused'
    (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        text,
        (w - w_text - 15, h_text + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
