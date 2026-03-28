import time

# Custom
from .BaseFPSCounter import BaseFPSCounter

class InstantFPSCounter(BaseFPSCounter):
    '''
    Calculates FPS from the time between two frames
    '''
    
    def __init__(self):
        # Initialize previous frame time

        self.prev_time = 0.0

    def update(self) -> float:
        # Return FPS for the current frame interval

        curr_time = time.time()
        delta = curr_time - self.prev_time
        
        fps = 1 / delta if delta > 0 else 0

        self.prev_time = curr_time

        return fps
    