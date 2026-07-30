"""Thin wrapper around a Whisper model + processor for transcription."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Fixed by Whisper's architecture (every checkpoint's feature extractor was trained on
# 16kHz mono audio) -- not something to source from config, unlike our own training
# pipeline's configurable sample_rate.
WHISPER_SAMPLE_RATE = 16_000


def resolve_model_path(config: dict) -> tuple[str, bool]:
    """Picks the fine-tuned model if it's present locally (real weight file, not
    just an empty directory), otherwise falls back to the pre-trained baseline.
    Returns (model_id_or_path, is_finetuned). Shared by the API and any script that
    should prefer the fine-tuned model when one's available."""
    finetuned_name = config["whisper"]["finetuned_model_name"]
    finetuned_dir = Path(config["paths"]["models_finetuned"]) / finetuned_name
    has_weights = any(finetuned_dir.glob("model.safetensors")) or any(
        finetuned_dir.glob("pytorch_model.bin")
    )
    if has_weights:
        return str(finetuned_dir), True
    return config["whisper"]["baseline_model"], False


class WhisperTranscriber:
    def __init__(self, model_id_or_path: str, language: str = "sw", task: str = "transcribe"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_id_or_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.language = language
        self.task = task

    def prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Our own training/eval data is always pre-resampled to 16kHz by
        # afrimeet.data.preprocess, so this never triggers there -- but arbitrary
        # uploads to the API (phones, recording/conferencing software) are commonly
        # 44.1kHz or 48kHz, and Whisper's feature extractor raises rather than
        # resampling automatically.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # downmix e.g. stereo to mono
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        return audio

    @torch.inference_mode()
    def _generate(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        audio = self.prepare_audio(audio, sample_rate)

        # truncation=False + padding="longest" + return_timestamps=True is Whisper's
        # own documented long-form transcription path (see
        # WhisperForConditionalGeneration.generate()'s docstring, "Longform
        # transcription"): audio over 30s is handled by Whisper's internal sequential
        # timestamp-based algorithm instead of being silently truncated to the first
        # 30s window (which a plain generate() call on fixed-size, padded/truncated
        # input_features does -- that was an earlier implementation, and it's why
        # transcriptions of real meeting-length audio were cutting off early). This
        # also works correctly for audio under 30s, so it's safe to use unconditionally
        # rather than branching on duration.
        #
        # NOTE: transformers.pipeline("automatic-speech-recognition", chunk_length_s=...)
        # looks like the obvious fix for long audio but is NOT the right tool here --
        # transformers itself logs a warning that chunk_length_s is "very experimental"
        # for seq2seq models like Whisper and recommends this generate()-based approach
        # instead, since Whisper has its own purpose-built long-form algorithm.
        inputs = self.processor(
            audio,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        ).to(self.device)

        return self.model.generate(
            **inputs,
            language=self.language,
            task=self.task,
            return_timestamps=True,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> str:
        predicted_ids = self._generate(audio, sample_rate)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()

    def transcribe_segments(
        self, audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE
    ) -> list[dict]:
        """Like transcribe(), but returns Whisper's own segment boundaries instead of one
        flat string: [{"start": float, "end": float, "text": str}, ...] (seconds). Used
        to auto-segment long audio (e.g. a meeting recording) into short clips for
        manual review/correction before adding it as training data -- Whisper predicts
        these boundaries as part of its long-form decoding algorithm; this just exposes
        them instead of discarding them the way transcribe() does."""
        predicted_ids = self._generate(audio, sample_rate)
        decoded = self.processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True, output_offsets=True
        )
        return [
            {"start": start, "end": end, "text": segment["text"].strip()}
            for segment in decoded["offsets"]
            for start, end in [segment["timestamp"]]
        ]
