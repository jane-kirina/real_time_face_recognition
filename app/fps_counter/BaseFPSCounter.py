from abc import ABC, abstractmethod

class BaseFPSCounter(ABC):
    @abstractmethod
    def update(self) -> float:
        pass