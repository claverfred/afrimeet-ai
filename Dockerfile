# AfriMeet AI API — Phase 6 deployment.
#
# Deliberately does NOT bake data/ or models/ into the image (multi-GB, and the
# fine-tuned model is produced separately by Colab training, see README.md) --
# mount them as volumes at runtime instead:
#
#   docker build -t afrimeet-ai-api .
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models" afrimeet-ai-api
#
# Falls back to the pre-trained baseline model if models/finetuned/ isn't mounted
# or is empty (see afrimeet.api.main.resolve_model_path).

FROM python:3.10-slim

# libsndfile1: required by soundfile; ffmpeg: broadens the audio formats accepted
# by /transcribe beyond what libsndfile alone decodes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir \
    -r requirements/base.txt \
    -r requirements/ml.txt \
    -r requirements/api.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

CMD ["uvicorn", "afrimeet.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
