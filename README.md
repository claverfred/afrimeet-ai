# AfriMeet AI

An intelligent multilingual meeting assistant with domain-adaptive speech recognition
for African institutions. MSc Data Science & AI research project — NM-AIST.

The core research contribution is fine-tuning OpenAI's **Whisper** ASR model on
multilingual African conference speech and evaluating whether domain adaptation
improves transcription accuracy over the pre-trained baseline. That model then powers
a meeting assistant: transcription, translation, diarization, summarization, action-item
extraction, and meeting search (RAG).

See [`docs/concept-note/`](docs/concept-note/) for the full concept note.

## Project structure

```
AfriMeet AI/
├── configs/                # YAML configs (paths, model/training hyperparameters)
├── data/
│   ├── raw/                 # Immutable original data (never edited)
│   ├── interim/              # Intermediate cleaned/segmented data
│   ├── processed/            # Final datasets ready for model training
│   └── external/              # Third-party reference data
├── docs/
│   └── concept-note/         # Original concept note (docx/pptx)
├── models/
│   ├── baseline/              # Baseline (pre-trained) Whisper artifacts/results
│   └── finetuned/               # Fine-tuned model checkpoints
├── notebooks/                # Exploratory analysis (naming: NN-initials-topic.ipynb)
├── reports/
│   ├── figures/                # Generated plots
│   └── metrics/                 # WER/CER/BLEU/ROUGE results, comparison tables
├── requirements/
│   ├── base.txt                 # Core lightweight deps
│   ├── ml.txt                    # Torch/transformers/datasets/ASR eval stack
│   ├── api.txt                    # FastAPI serving stack
│   └── dev.txt                     # Test/lint/format tooling
├── scripts/                  # Thin CLI entry points that call into src/afrimeet
├── src/afrimeet/             # Installable Python package (the actual pipeline code)
│   ├── data/                  # Download & preprocessing (Phase 2)
│   ├── models/                 # Training, evaluation, inference (Phases 3-5)
│   ├── translation/             # Kiswahili <-> English NMT
│   ├── diarization/              # Speaker diarization
│   ├── summarization/             # LLM summarization, minutes, action items
│   ├── api/                        # FastAPI app (Phase 6)
│   └── utils/                       # Shared helpers (config/logging)
├── tests/                    # Unit tests (pytest)
├── web/                      # Frontend (React/Next.js) — added in a later phase
├── .env.example              # Template for local secrets/config
├── pyproject.toml            # Package metadata + tool config (black/ruff/isort/pytest)
└── requirements.txt           # Convenience: installs everything
```

This layout follows standard data-science project conventions (raw data is
never mutated in place; code, data, and generated artifacts are kept in separate
trees) so the pipeline is reproducible end-to-end from a clean checkout.

## Setup

Requires Python 3.10 or 3.11 (PyTorch/Whisper compatibility; Python 3.13+ is not yet
well supported by some ML deps).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements/base.txt -r requirements/dev.txt   # lightweight, always
pip install -r requirements/ml.txt                              # heavy, needed for ASR work
pip install -r requirements/api.txt                              # needed to run the API

# 3. Install the local package in editable mode
pip install -e .

# 4. Copy the env template and fill in any secrets you need
cp .env.example .env
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first (see
https://pytorch.org/get-started/locally/) before installing `requirements/ml.txt`,
so pip doesn't fall back to the CPU wheel.

## Methodology (project phases)

1. **Problem definition** — done, see concept note.
2. **Data collection & preparation** — `src/afrimeet/data/` (`scripts/download_data.py`,
   `scripts/prepare_dataset.py`).
3. **Baseline modeling** — evaluate pre-trained Whisper (WER/CER) —
   `scripts/evaluate.py --model openai/whisper-small --run-name baseline`.
4. **Model improvement** — fine-tune Whisper on conference-domain speech —
   `scripts/train.py`.
5. **Model evaluation** — compare baseline vs. fine-tuned (WER, CER, latency) —
   `scripts/evaluate.py --model models/finetuned/... --run-name finetuned` then
   `scripts/compare_models.py --runs baseline=... finetuned=...`.
6. **Deployment** — REST API (`src/afrimeet/api/`), web app (`web/`), live demo.

## Hybrid local / Colab workflow

Fine-tuning Whisper on CPU is impractically slow, so GPU-heavy phases run on Colab
while data download/prep and everyday development stay local:

| Phase | Where | Why |
|---|---|---|
| 2. Data collection & prep | Local (or Colab) | Bandwidth-bound, not compute-bound |
| 3. Baseline evaluation | Colab (GPU) | Fast batch inference |
| 4. Fine-tuning | Colab (GPU) | Impractical on CPU |
| 5. Comparison evaluation | Colab (GPU) | Same as Phase 3 |
| 6. API / deployment | Local / cloud | Not GPU-bound |

To run Phases 3-5 on Colab:

1. Open `notebooks/finetune_whisper_colab.ipynb` in Colab (`Runtime -> Change runtime
   type -> GPU`) and run the cells top to bottom.
2. The code is pulled from this GitHub repo (`REPO_URL` in the config cell). Since the
   repo is private, the clone cell prompts for a GitHub Personal Access Token
   (`repo` scope, or a fine-grained token scoped to just this repo) via `getpass` — it's
   only held in memory for that cell, never written to disk or saved in the notebook.

If you'd rather not use a token, clear `REPO_URL` in the notebook to fall back to a
Drive-zip instead: run `python scripts/package_for_colab.py` locally (writes
`dist/afrimeet-ai-code.zip`, excludes data/models/venv) and upload it to
`My Drive/AfriMeet_AI/afrimeet-ai-code.zip`.

The notebook doesn't duplicate any pipeline logic — it clones the same `afrimeet`
package and calls the same `scripts/` used locally. Google Drive persists the
processed dataset and model checkpoints between sessions (Colab's local disk is wiped
on disconnect); see the notebook's markdown cells for how caching and resuming work.

## Running tests

```bash
pytest
```

## Status

Project scaffold, data pipeline, and baseline evaluation are in progress. See
open todos in the repository issue tracker / commit history for current phase.
