import time

# Custom
from .BaseFPSCounter import BaseFPSCounter

class InstantFPSCounter(BaseFPSCounter):
    def __init__(self):
        self.prev_time = 0.0

    def update(self) -> float:
        curr_time = time.time()
        delta = curr_time - self.prev_time
        
        fps = 1 / delta if delta > 0 else 0

        self.prev_time = curr_time

        return fps
    