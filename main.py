# Custom
from app.fps_counter.AverageFPSCounter import AverageFPSCounter
from app.fps_counter.InstantFPSCounter import InstantFPSCounter
from app.fps_counter.SmoothedFPSCounter import SmoothedFPSCounter
from app.logger import EventLogger

from app.detector import init_face_detector
from app.webcam import start_camera


def main():
    logger = EventLogger()

    fps_counter_avg = AverageFPSCounter(interval=1.0)
    fps_counter_inst = InstantFPSCounter()
    fps_counter_smooth = SmoothedFPSCounter(alpha=0.5)

    face_detector = init_face_detector(model_name='buffalo_s', det_size=(320, 320))
    
    logger.log_system('FACE_DETECTOR_INIT', model='buffalo_s', det_size=(320, 320))

    start_camera(fps_counter_avg, face_detector, logger, scale=0.75, match_threshold = 0.5, detect_every_n_frames=4)

if __name__ == "__main__":
    main()