"""Thin wrapper around a Whisper model + processor for transcription."""

from __future__ import annotations

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Fixed by Whisper's architecture (every checkpoint's feature extractor was trained on
# 16kHz mono audio) -- not something to source from config, unlike our own training
# pipeline's configurable sample_rate.
WHISPER_SAMPLE_RATE = 16_000


class WhisperTranscriber:
    def __init__(self, model_id_or_path: str, language: str = "sw", task: str = "transcribe"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_id_or_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> str:
        # Our own training/eval data is always pre-resampled to 16kHz by
        # afrimeet.data.preprocess, so this never triggers there -- but arbitrary
        # uploads to the API (phones, recording/conferencing software) are commonly
        # 44.1kHz or 48kHz, and Whisper's feature extractor raises rather than
        # resampling automatically.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # downmix e.g. stereo to mono
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
            sample_rate = WHISPER_SAMPLE_RATE

        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)
        predicted_ids = self.model.generate(inputs, forced_decoder_ids=self.forced_decoder_ids)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()
