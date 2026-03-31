import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List
import uvicorn

# Custom
from app.embedding import (load_db, save_db)
from app.offline_pipeline import enroll_from_uploads

app = FastAPI(
    title='Face Recognition API',
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json'
)

# Main page
@app.get('/')
def root():
    return {'message': 'Face Recognition API is running'}

# TODO Health page
@app.get('/health')
def health():
    return {'status': 'ok'}

# List with embeddings + names
@app.get('/persons')
def get_persons():
    db = load_db()
    return {
        'total_persons': len(db),
        'persons': [
            {'name': name, 'embeddings_count': len(embeddings)}
            for name, embeddings in sorted(db.items())
        ]
    }

# Page with logs
@app.get('/events')
def get_events(limit: int = 40):
    path = 'data/events.jsonl'

    if not os.path.exists(path):
        return {'total': 0, 'events': []}

    events = []
    with open(path, 'r', encoding='utf-8') as f:
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

# Add new 
@app.post('/enroll')
def enroll(
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        if not files:
            raise ValueError('No files uploaded')

        result = enroll_from_uploads(name, files)
        result['note'] = 'Restart camera to reload face_db and FAISS index'
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Delete person by name
@app.delete('/persons/{name}')
def delete_person(name):
    db = load_db()

    if name not in db:
        raise HTTPException(status_code=404, detail='Person not found')

    del db[name]
    save_db(db)

    return {
        'status': 'deleted',
        'name': name,
        'note': 'Restart camera to reload face_db and FAISS index'
    }
