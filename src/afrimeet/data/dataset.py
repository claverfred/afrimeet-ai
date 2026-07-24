"""PyTorch Dataset over a processed manifest.csv (used for both baseline
evaluation and fine-tuning)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset


class ManifestAudioDataset(Dataset):
    """Loads (audio_array, sample_rate, transcript) triples from a manifest.csv
    produced by `afrimeet.data.preprocess`."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.df = pd.read_csv(self.manifest_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = self.base_dir / row["audio_path"]
        signal, sample_rate = sf.read(audio_path, dtype="float32")
        return {
            "audio": signal,
            "sample_rate": sample_rate,
            "transcript": row["transcript"],
            "audio_path": str(audio_path),
        }

    @classmethod
    def from_multiple(cls, manifest_paths: list[str | Path]) -> pd.DataFrame:
        """Concatenate several manifests' underlying DataFrames with an added
        `source_dir` column (useful for building a single train set across
        multiple raw datasets)."""
        frames = []
        for path in manifest_paths:
            path = Path(path)
            df = pd.read_csv(path)
            df["source_dir"] = str(path.parent)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
