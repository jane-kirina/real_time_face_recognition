import cv2

# Util func
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

# Webcam stream, video loop with fps
def start_camera(fps_counter):
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        raise RuntimeError('No stream: Could not open webcam')
    
    window_name = 'Webcam - Live'

    while True:
        ret, frame = stream.read()

        if not ret:
            print('Failed to read frame from webcam')
            # raise RuntimeError('Failed to read frame from webcam') # TODO
            break
        
        fps = fps_counter.update()
        draw_fps(frame, fps)

        cv2.imshow(window_name, frame)

        # to exit stream
        key = cv2.waitKey(1)

        if key == ord('q') or key == ord('й'):
            break

    stream.release()
    cv2.destroyAllWindows()
