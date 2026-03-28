import time

# Custom
from .BaseFPSCounter import BaseFPSCounter

class AverageFPSCounter(BaseFPSCounter):
    '''
    Calculates the average FPS for a certain time (default is 1 sec)
    '''

    def __init__(self, interval: float = 1.0):
        # Initialize counter with update interval in seconds

        self.interval = interval
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        # Update frame count and return average FPS

        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= self.interval:
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps
