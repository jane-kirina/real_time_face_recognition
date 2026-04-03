# Custom
from app.fps_counter.AverageFPSCounter import AverageFPSCounter
from app.logger import EventLogger

from app.detector import init_face_detector
from app.webcam import start_camera
from app.camera_state import CameraState

from app.config import settings

def main():
    logger = EventLogger()
    state = CameraState(logger=logger, scale = 0.5)
    fps_counter_avg = AverageFPSCounter()

    model_name=settings.model_name
    det_size=(settings.det_size, settings.det_size)
    match_threshold = settings.match_threshold
    detect_every_n_frames=settings.detect_every_n_frames

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