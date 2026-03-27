# Custom
from app.fps_counter.AverageFPSCounter import AverageFPSCounter
from app.fps_counter.InstantFPSCounter import InstantFPSCounter
from app.fps_counter.SmoothedFPSCounter import SmoothedFPSCounter
from app.webcam import start_camera

if __name__ == "__main__":
    fps_counter_avg = AverageFPSCounter(interval=1.0)
    fps_counter_inst = InstantFPSCounter()
    fps_counter_smooth = SmoothedFPSCounter(alpha=0.5)

    start_camera(fps_counter_smooth)