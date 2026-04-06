# TODO TEST EVERYTHING
import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List
import uvicorn
from pydantic import BaseModel
import numpy as np
from contextlib import asynccontextmanager

# Custom
from app.config import settings
from app.registry import FaceRegistry

app = FastAPI(
    title='Face Recognition API',
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json'
)

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
    lifespan=lifespan
)

class EnrollRequest(BaseModel):
    name: str
    embedding: list[float]


# Main page
@app.get('/')
def root():
    return {'message': 'Face Recognition API is running'}

# TODO Health page
@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.get('/stats')
def stats():
    return registry.stats()

# List with embeddings + names
@app.get('/persons')
def get_persons():
    return {
        'persons': registry.labels,
        'count': len(registry.labels),
        'version': registry.version
    }

# Page with logs
@app.get('/events') # TODO
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

# Reload FAISS index
@app.post('/reload-index')
def reload_index():
    registry.reload()
    return {
        'status': 'ok',
        'message': 'Index reloaded',
        'stats': registry.stats()
    }

# Add new 
@app.post('/enroll')
def enroll(req: EnrollRequest):
    emb = np.array(req.embedding, dtype=np.float32)
    registry.add_person(req.name, emb)

    return {
        'status': 'ok',
        'message': f'{req.name} added',
        'stats': registry.stats()
    }

# Delete person by name
@app.delete('/persons/{name}')
def delete_person(name):
    deleted = registry.delete_person(name)

    if deleted == 0:
        raise HTTPException(404, 'Person not found')

    return {
        'status': 'ok',
        'deleted': deleted,
        'stats': registry.stats()
    }
