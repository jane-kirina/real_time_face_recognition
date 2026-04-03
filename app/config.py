from pydantic import Field
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

    # App
    app_name: str = 'Real-Time Face Recognition'
    debug: bool = False
    log_level: str = 'INFO'

    # Recognition
    model_name: str = 'buffalo_sc'
    det_size: int = 320
    match_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    detect_every_n_frames: int = Field(default=3, ge=1)

    # Tracking (smoothing)
    track_max_distance: int = 80
    track_max_missed: int = 10
    smoothing_window: int = 5
    log_cooldown_sec: int = 5

    # Paths
    face_db_path: Path = Path('data/face_db.npy')
    events_log_path: Path = Path('data/events.jsonl')

    # API
    api_application: str = 'app.api:app'
    api_host: str = '127.0.0.1'
    api_port: int = 8000
    api_reload: bool = True

settings = Settings()