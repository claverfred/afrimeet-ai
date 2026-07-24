"""Phase 6 — minimal REST API exposing the ASR model.

Run with: uvicorn afrimeet.api.main:app --reload
Requires requirements/ml.txt and requirements/api.txt to be installed.

This is intentionally small: one health check and one transcription endpoint.
Translation, diarization, summarization, and RAG search are separate modules
(see src/afrimeet/translation, diarization, summarization) to be wired in here
as they're built out.
"""

from __future__ import annotations

import io
from functools import lru_cache

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile

from afrimeet.models.inference import WhisperTranscriber
from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger

app = FastAPI(
    title="AfriMeet AI API",
    description="Multilingual meeting assistant with domain-adaptive speech recognition.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_transcriber() -> WhisperTranscriber:
    config = load_config()
    model_id = config["whisper"]["baseline_model"]
    language = config["whisper"]["language"]
    logger.info(f"Loading ASR model for API: {model_id}")
    return WhisperTranscriber(model_id, language=language)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile) -> dict:
    if file.content_type not in (None, "") and not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Expected an audio file upload.")

    raw_bytes = await file.read()
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    transcriber = get_transcriber()
    text = transcriber.transcribe(audio, sample_rate)
    return {"transcript": text}
