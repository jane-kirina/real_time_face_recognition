import time
import json

class EventLogger:
    def __init__(self, cooldown=5.0, log_file='events.jsonl'):
        # cooldown: seconds before the same identity can be logged again
        self.cooldown = cooldown
        self.log_file = log_file
        self.last_logged = {}
    
    # ----------------------------
    # System logs
    # ----------------------------
    def log_system(self, event_type, **kwargs):
        event = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'event': event_type,
            **kwargs
        }
        self.write_event(event)

    def write_event(self, event: dict):
        print(event)
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(event, ensure_ascii=False) + '\n')

    # ----------------------------
    # Detection logs
    # ----------------------------
    def log_detection(self, name, face_id=None, confidence=None):
        now = time.time()

        # normalize identity key
        identity = face_id  if face_id is not None else name

        # cooldown check
        last_time = self.last_logged.get(identity, 0)
        if now - last_time < self.cooldown:
            return  # skip spam

        # update last log time
        self.last_logged[identity] = now

        # choose event type
        if name == 'unknown':
            event_type = 'UNKNOWN_DETECTED'
        else:
            event_type = "PERSON_DETECTED"

        event = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'event': event_type,
            'name': name,
            'confidence': float(confidence) if confidence is not None else None
        }

        self.write_event(event)
