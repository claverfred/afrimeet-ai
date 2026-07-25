"""Phase 4 — Fine-tune Whisper on domain-specific (conference) speech.

Uses HF `Seq2SeqTrainer` on top of a manifest-based dataset. Hyperparameters
(learning rate, batch size, epochs, ...) come from `configs/config.yaml`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate as hf_evaluate
import torch
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from afrimeet.utils.config import load_config
from afrimeet.utils.logging import logger

CHECKPOINT_BACKUP_ENV_VAR = "AFRIMEET_CHECKPOINT_BACKUP_DIR"


class CheckpointBackupCallback(TrainerCallback):
    """Copies the checkpoint just saved by the Trainer to a persistent backup
    directory (e.g. a Google Drive mount in Colab), overwriting the previous
    backup each time. Protects long unattended runs against the training VM's
    local disk being lost to a disconnect — see `main()` for how the backup is
    also used to auto-resume."""

    def __init__(self, backup_dir: str):
        self.backup_dir = Path(backup_dir)

    def on_save(self, args, state, control, **kwargs):  # noqa: D102
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return
        target = self.backup_dir / "latest_checkpoint"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint_dir, target)
        logger.info(f"Backed up checkpoint at step {state.global_step} to {target}")


def _load_split_as_hf_dataset(manifest_path: Path, sample_rate: int) -> Dataset:
    """Builds a Dataset that only holds file paths + transcripts in memory. Audio
    is decoded lazily (per-example, on access) via the `Audio` feature cast below --
    eagerly loading every waveform into a Python list upfront (the previous approach)
    holds the whole split in RAM simultaneously, which OOM-kills the process on any
    dataset larger than a toy one (this is what was killing training with SIGKILL)."""
    import pandas as pd

    df = pd.read_csv(manifest_path)
    base_dir = manifest_path.parent

    audio_paths = [str(base_dir / p) for p in df["audio_path"]]
    dataset = Dataset.from_dict({"audio": audio_paths, "transcript": df["transcript"].tolist()})
    return dataset.cast_column("audio", Audio(sampling_rate=sample_rate))


def build_dataset(
    processed_dir: Path,
    sample_rate: int,
    train_split: str = "train",
    eval_split: str = "test",
) -> DatasetDict:
    """Collects every dataset under `data/processed/*/train` (and `*/test`) into a
    single combined HF DatasetDict."""
    train_manifests = sorted(processed_dir.glob(f"*/{train_split}/manifest.csv"))
    eval_manifests = sorted(processed_dir.glob(f"*/{eval_split}/manifest.csv"))

    if not train_manifests:
        raise FileNotFoundError(
            f"No training manifests found under {processed_dir}/*/{train_split}/manifest.csv "
            "— run scripts/download_data.py and scripts/prepare_dataset.py first."
        )

    from datasets import concatenate_datasets

    train_ds = concatenate_datasets(
        [_load_split_as_hf_dataset(p, sample_rate) for p in train_manifests]
    )
    eval_ds = (
        concatenate_datasets([_load_split_as_hf_dataset(p, sample_rate) for p in eval_manifests])
        if eval_manifests
        else train_ds.select(range(min(50, len(train_ds))))
    )
    return DatasetDict(train=train_ds, test=eval_ds)


@dataclass
class WhisperDataCollator:
    """Converts raw audio to Whisper's mel-spectrogram `input_features` here, per
    batch, rather than eagerly for the whole dataset up front. Whisper's feature
    extractor pads every clip to a fixed 30s window regardless of its actual length,
    so each spectrogram is a fixed ~960KB -- pre-computing that for tens of thousands
    of examples via `dataset.map()` produces tens of GB and reliably OOM-kills the
    process before training even starts. Doing it per-batch keeps memory bounded to
    O(batch_size) regardless of dataset size."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {
                "input_features": self.processor.feature_extractor(
                    f["audio"]["array"], sampling_rate=f["audio"]["sampling_rate"]
                ).input_features[0]
            }
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_labels(dataset: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
    """Tokenizes transcripts into `labels` (cheap, text-only). Audio stays as the lazy
    `Audio` feature set up in `_load_split_as_hf_dataset` and is only decoded +
    converted to spectrogram features inside `WhisperDataCollator`, per batch."""

    def _tokenize(example):
        example["labels"] = processor.tokenizer(example["transcript"]).input_ids
        return example

    return dataset.map(_tokenize, remove_columns=["transcript"])


def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on conference speech.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-model", default=None, help="Override config whisper.baseline_model")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    processed_dir = Path(config["paths"]["data_processed"])
    finetuned_name = config["whisper"]["finetuned_model_name"]
    output_dir = Path(config["paths"]["models_finetuned"]) / finetuned_name
    base_model = args.base_model or config["whisper"]["baseline_model"]
    language = config["whisper"]["language"]
    task = config["whisper"]["task"]
    train_cfg = config["training"]

    logger.info(f"Base model: {base_model}")
    dataset = build_dataset(processed_dir, sample_rate=config["data"]["sample_rate"])
    logger.info(f"Train examples: {len(dataset['train'])}, eval examples: {len(dataset['test'])}")

    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    dataset = prepare_labels(dataset, processor)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    data_collator = WhisperDataCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        fp16=train_cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 2),
        logging_steps=train_cfg["logging_steps"],
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        # Required whenever a custom collator does its own preprocessing from a raw
        # column: Trainer's default column-pruning drops any dataset column that isn't
        # a named parameter of the model's forward() -- "audio" isn't (only the
        # *converted* "input_features" is), so it was getting silently stripped before
        # WhisperDataCollator ever saw it, surfacing as a KeyError('audio') mid-training.
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # If AFRIMEET_CHECKPOINT_BACKUP_DIR is set (the Colab notebook sets it to a Drive
    # path), back up each checkpoint there as training progresses, and resume from it
    # automatically if a backup from an interrupted run is already present -- this is
    # what protects a long unattended run against the training VM being reclaimed
    # mid-run (its local disk doesn't survive that, only the Drive backup does).
    callbacks = []
    resume_from_checkpoint = None
    backup_dir = os.environ.get(CHECKPOINT_BACKUP_ENV_VAR)
    if backup_dir:
        callbacks.append(CheckpointBackupCallback(backup_dir))
        candidate = Path(backup_dir) / "latest_checkpoint"
        if candidate.exists():
            resume_from_checkpoint = str(candidate)
            logger.info(f"Found a checkpoint backup at {candidate} — resuming training from it")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks or None,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"Fine-tuned model saved to {output_dir}")

    if backup_dir:
        # Training finished cleanly -- the mid-run backup has served its purpose and
        # would otherwise incorrectly trigger a resume on the *next* fresh training run.
        shutil.rmtree(Path(backup_dir) / "latest_checkpoint", ignore_errors=True)


if __name__ == "__main__":
    main()
