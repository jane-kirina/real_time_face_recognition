from dataclasses import dataclass, field
from typing import Any

@dataclass
class CameraState:
    # Keep information about one frame: frame itself, frame_id, FPS, etc.
    paused: bool = False
    frame: Any = None
    fps: float = 0.0
    scale: float = 0.75
    frame_id: int = 0
    faces: list = field(default_factory=list)
    display_frame: Any = None
    db: dict = field(default_factory=dict)
    faiss_index: Any = None
    faiss_names: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    next_track_id: int = 1
    logger: Any = None