# Custom
from app.fps_counter.AverageFPSCounter import AverageFPSCounter
from app.fps_counter.InstantFPSCounter import InstantFPSCounter
from app.fps_counter.SmoothedFPSCounter import SmoothedFPSCounter

from app.detector import init_face_detector

from app.webcam import start_camera


def main():
    fps_counter_avg = AverageFPSCounter(interval=1.0)
    fps_counter_inst = InstantFPSCounter()
    fps_counter_smooth = SmoothedFPSCounter(alpha=0.5)

    face_detector = init_face_detector(model_name='buffalo_sc', det_size=(160, 160))
    
    start_camera(fps_counter_avg, face_detector, scale=0.3, match_threshold = 0.6)

if __name__ == "__main__":
    main()