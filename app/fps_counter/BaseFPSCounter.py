from abc import ABC, abstractmethod

class BaseFPSCounter(ABC):
    '''
    Base abstract class for FPS counters
    '''

    @abstractmethod
    def update(self) -> float:
        # Update counter state and return current FPS value
        
        pass