import time

# Custom
from .BaseFPSCounter import BaseFPSCounter

class SmoothedFPSCounter(BaseFPSCounter):
    def __init__(self, alpha: float = 0.1):
        self.prev_time = 0.0
        self.fps = 0.0
        self.alpha = alpha

    def update(self) -> float:
        curr_time = time.time()
        delta = curr_time - self.prev_time

        instant_fps = 1 / delta if delta > 0 else 0

        if self.fps > 0:
            self.fps = (1 - self.alpha) * self.fps + self.alpha * instant_fps
        else:
            self.fps = instant_fps

        self.prev_time = curr_time
        return self.fps
    