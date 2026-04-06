import time
from app.logger import EventLogger

def test_log_detection_calls_write_event(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_detection('alice', face_id=1, score=0.9)

    assert len(called) == 1
    assert called[0]['event'] == 'FACE_RECOGNIZED'
    assert called[0]['name'] == 'alice'
    assert called[0]['score'] == 0.9

def test_log_detection_unknown_event(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_detection('unknown', face_id=1, score=0.3)

    assert len(called) == 1
    assert called[0]['event'] == 'UNKNOWN_FACE_DETECTED'
    assert called[0]['name'] == 'unknown'

def test_log_detection_respects_cooldown(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_detection('alice', face_id=1, score=0.9)
    logger.log_detection('alice', face_id=1, score=0.95)

    # second call should be skipped
    assert len(called) == 1

def test_log_detection_allows_after_cooldown(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_detection('alice', face_id=1, score=0.9)

    # simulate time passing
    logger.last_logged[1] -= 6

    logger.log_detection('alice', face_id=1, score=0.95)

    assert len(called) == 2

def test_log_detection_uses_face_id_as_identity(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    # same name, different face_id should log twice
    logger.log_detection('alice', face_id=1, score=0.9)
    logger.log_detection('alice', face_id=2, score=0.9)

    assert len(called) == 2

def test_log_detection_uses_name_if_no_face_id(monkeypatch):
    logger = EventLogger(cooldown=5.0)

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_detection('alice', face_id=None, score=0.9)
    logger.log_detection('alice', face_id=None, score=0.95)

    # cooldown applies when same name
    assert len(called) == 1

def test_log_system_calls_write_event(monkeypatch):
    logger = EventLogger()

    called = []

    def fake_write(event):
        called.append(event)

    monkeypatch.setattr(logger, 'write_event', fake_write)

    logger.log_system('CAMERA_STARTED', device=0)

    assert len(called) == 1
    assert called[0]['event'] == 'CAMERA_STARTED'
    assert called[0]['device'] == 0
