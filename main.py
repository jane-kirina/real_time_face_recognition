# Custom
from app.fps_counter.AverageFPSCounter import AverageFPSCounter
from app.logger import EventLogger

from app.detector import init_face_detector
from app.webcam import start_camera
from app.CameraState import CameraState

def main():
    logger = EventLogger()
    state = CameraState(logger=logger, scale = 0.5)
    fps_counter_avg = AverageFPSCounter(interval=1.0)

    model_name='buffalo_s'
    det_size=(320, 320)
    match_threshold = 0.5
    detect_every_n_frames=4

    face_detector = init_face_detector(model_name=model_name, det_size=det_size)
    
    logger.log_system('FACE_DETECTOR_INIT', model=model_name, det_size=det_size)

    state.logger.log_system('START_CAMERA', 
                                      scale=state.scale, 
                                      match_threshold = match_threshold, 
                                      detect_every_n_frames=detect_every_n_frames)

    start_camera(fps_counter_avg, 
                 face_detector, 
                 state, 
                 match_threshold = match_threshold, 
                 detect_every_n_frames=detect_every_n_frames)

if __name__ == "__main__":
    main()