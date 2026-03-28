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
    face_detector = init_face_detector() # TODO try buffalo_sc model
    start_camera(fps_counter_avg, face_detector)

if __name__ == "__main__":
    main()