import types
from pathlib import Path
import pytest
import json
from fastapi.testclient import TestClient
import app.api as api_module

class FakeRegistry:
    def __init__(self):
        self.labels = ['alice', 'bob']
        self.version = 3
        self.loaded = False
        self.reloaded = False
        self.added = []
        self.deleted = {}

    def load(self):
        self.loaded = True

    def reload(self):
        self.reloaded = True

    def stats(self):
        return {
            'count': len(self.labels),
            'version': self.version,
        }

    def add_person(self, name, emb):
        self.added.append((name, emb))
        self.labels.append(name)
        self.version += 1

    def delete_person(self, name):
        if name in self.labels:
            self.labels.remove(name)
            self.version += 1
            return 1
        return 0

def make_client(monkeypatch, tmp_path):
    fake_registry = FakeRegistry()
    monkeypatch.setattr(api_module, 'registry', fake_registry)

    events_file = tmp_path / 'events.jsonl'
    monkeypatch.setattr(api_module.os.path, 'exists', lambda path: path == 'data/events.jsonl')

    original_open = open

    def fake_open(path, mode='r', encoding=None):
        if path == 'data/events.jsonl':
            return original_open(events_file, mode, encoding=encoding)
        return original_open(path, mode, encoding=encoding)

    monkeypatch.setattr(api_module, 'open', fake_open, raising=False)

    client = TestClient(api_module.app)
    return client, fake_registry, events_file

def test_root_endpoint(monkeypatch, tmp_path):
    client, _, _ = make_client(monkeypatch, tmp_path)

    response = client.get('/')

    assert response.status_code == 200
    assert response.json() == {'message': 'Face Recognition API is running'}

def test_health_endpoint(monkeypatch, tmp_path):
    client, _, _ = make_client(monkeypatch, tmp_path)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}

def test_stats_endpoint(monkeypatch, tmp_path):
    client, fake_registry, _ = make_client(monkeypatch, tmp_path)

    response = client.get('/stats')

    assert response.status_code == 200
    assert response.json() == fake_registry.stats()

def test_persons_endpoint_returns_labels_count_and_version(monkeypatch, tmp_path):
    client, fake_registry, _ = make_client(monkeypatch, tmp_path)

    response = client.get('/persons')

    assert response.status_code == 200
    assert response.json() == {
        'persons': fake_registry.labels,
        'count': len(fake_registry.labels),
        'version': fake_registry.version
    }

def test_events_endpoint_returns_empty_when_file_missing(monkeypatch, tmp_path):
    fake_registry = FakeRegistry()
    monkeypatch.setattr(api_module, 'registry', fake_registry)
    monkeypatch.setattr(api_module.os.path, 'exists', lambda path: False)

    client = TestClient(api_module.app)
    response = client.get('/events')

    assert response.status_code == 200
    assert response.json() == {'total': 0, 'events': []}

def test_events_endpoint_returns_last_n_events(monkeypatch, tmp_path):
    client, _, events_file = make_client(monkeypatch, tmp_path)

    events = [
        {'event': 'PERSON_DETECTED', 'name': 'alice'},
        {'event': 'UNKNOWN_DETECTED', 'name': 'unknown'},
        {'event': 'PERSON_DETECTED', 'name': 'bob'}
    ]
    with open(events_file, 'w', encoding='utf-8') as f:
        for item in events:
            f.write(json.dumps(item) + '\n')

    response = client.get('/events?limit=2')

    assert response.status_code == 200
    assert response.json() == {
        'total': 3,
        'events': events[-2:]
    }

def test_reload_index_endpoint(monkeypatch, tmp_path):
    client, fake_registry, _ = make_client(monkeypatch, tmp_path)

    response = client.post('/reload-index')

    assert response.status_code == 200
    assert fake_registry.reloaded is True
    assert response.json() == {
        'status': 'ok',
        'message': 'Index reloaded',
        'stats': fake_registry.stats()
    }

def test_enroll_endpoint_adds_person(monkeypatch, tmp_path):
    client, fake_registry, _ = make_client(monkeypatch, tmp_path)

    payload = {
        'name': 'charlie',
        'embedding': [0.1, 0.2, 0.3]
    }

    response = client.post('/enroll', json=payload)

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'
    assert response.json()['message'] == 'charlie added'
    assert fake_registry.added[0][0] == 'charlie'
    assert fake_registry.added[0][1].dtype.name == 'float32'

def test_delete_person_endpoint_returns_404_for_missing_person(monkeypatch, tmp_path):
    client, _, _ = make_client(monkeypatch, tmp_path)

    response = client.delete('/persons/not-found')

    assert response.status_code == 404
    assert response.json()['detail'] == 'Person not found'

def test_delete_person_endpoint_deletes_existing_person(monkeypatch, tmp_path):
    client, fake_registry, _ = make_client(monkeypatch, tmp_path)

    response = client.delete('/persons/alice')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'ok',
        'deleted': 1,
        'stats': fake_registry.stats()
    }
    assert 'alice' not in fake_registry.labels