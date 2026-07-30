"""Phase 6 — minimal REST API exposing the ASR model.

Run with: uvicorn afrimeet.api.main:app --reload
Requires requirements/ml.txt and requirements/api.txt to be installed. The
fine-tuned model produced by Phase 4 (trained in Colab) isn't in this repo --
see README.md for how to get it from Google Drive onto whatever machine runs
this API.

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

from afrimeet.models.inference import WhisperTranscriber, resolve_model_path
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
    model_id, is_finetuned = resolve_model_path(config)
    language = config["whisper"]["language"]
    logger.info(f"Loading ASR model for API: {model_id} (fine-tuned: {is_finetuned})")
    return WhisperTranscriber(model_id, language=language)


@app.get("/health")
def health() -> dict:
    config = load_config()
    model_id, is_finetuned = resolve_model_path(config)
    return {"status": "ok", "model": model_id, "finetuned": is_finetuned}


@app.post("/transcribe")
async def transcribe(file: UploadFile) -> dict:
    # No content_type gate here: many HTTP clients send application/octet-stream (or
    # omit it) for perfectly valid audio, which caused real uploads to be rejected.
    # sf.read() below is the actual, authoritative validation -- it decodes real audio
    # and cleanly rejects anything that isn't, so a separate content_type pre-check is
    # both redundant and a source of false rejections.
    raw_bytes = await file.read()
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    transcriber = get_transcriber()
    text = transcriber.transcribe(audio, sample_rate)
    return {"transcript": text}
