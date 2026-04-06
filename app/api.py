import os
import json
from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
from pydantic import BaseModel
import numpy as np
from contextlib import asynccontextmanager

# Custom
from app.config import settings
from app.registry import FaceRegistry

registry = FaceRegistry(settings.face_db_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        registry.load()
        print('Registry loaded')
        yield
    finally:
        print('App shutdown')

app = FastAPI(
    title='Face Recognition API',
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json',
    lifespan=lifespan
)


# ----------------------------
# Models
# ----------------------------
class EnrollRequest(BaseModel):
    name: str
    embedding: list[float]


# ----------------------------
# Core endpoints
# ----------------------------
@app.get('/')
def root():
    return {'message': 'Face Recognition API is running'}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/stats')
def stats():
    return registry.stats()

@app.get('/persons')
def get_persons():
    return {
        'persons': registry.labels,
        'count': len(registry.labels),
        'version': registry.version
    }

# ----------------------------
# Events / Logs
# ----------------------------
@app.get('/events')
def get_events(limit: int = 40):
    if not os.path.exists(settings.events_log_path):
        return {'total': 0, 'events': []}

    events = []
    with open(settings.events_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return {
        'total': len(events),
        'events': events[-limit:]
    }

# ----------------------------
# Registry actions
# ----------------------------
@app.post('/index/index')
def reload_index():
    registry.reload()
    return {
        'status': 'ok',
        'message': 'Index reloaded successfully',
        'stats': registry.stats()
    }

@app.post('/person')
def enroll_person(req: EnrollRequest):
    emb = np.array(req.embedding, dtype=np.float32)
    registry.add_person(req.name, emb)

    return {
        'status': 'ok',
        'message': f"Person '{req.name}' enrolled successfully",
        'stats': registry.stats()
    }


@app.delete('/persons/{name}')
def delete_person(name):
    deleted = registry.delete_person(name)

    if deleted == 0:
        raise HTTPException(404, 'Person not found')

    return {
        'status': 'ok',
        'message': f"Person '{name}' deleted",
        'deleted': deleted,
        'stats': registry.stats()
    }
