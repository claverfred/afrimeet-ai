"""Thin wrapper around a Whisper model + processor for transcription."""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


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
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)
        predicted_ids = self.model.generate(inputs, forced_decoder_ids=self.forced_decoder_ids)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()
