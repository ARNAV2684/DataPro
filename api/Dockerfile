# Garuda ML Pipeline — Backend (FastAPI + numeric/text/image pipeline scripts)
#
# Build context MUST be the repository root, because the API runs the pipeline
# scripts (2and3week/, 4and5week/, EDA/, 6week/, imagelib/) as subprocesses
# resolved relative to the repo root:
#
#   docker build -f api/Dockerfile -t garuda-backend:local .
#
FROM python:3.11-slim

# System libraries required by Pillow, matplotlib and torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids pulling multi-GB CUDA wheels), then the
# rest of the backend + model-script dependencies.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 torchvision==0.16.0 \
    && pip install --no-cache-dir -r /app/api/requirements.txt \
    && pip install --no-cache-dir 'xgboost>=1.7.0' 'seaborn>=0.12.0' 'joblib>=1.3.0' \
        'tqdm>=4.65.0' 'datasets>=2.12.0'

# Copy the full project (API + pipeline scripts + imagelib). See .dockerignore.
COPY . /app

# The API imports its sibling packages (routes/, shared/, models/) as top-level
# modules, so it must run from inside api/. module_runner then resolves the
# workspace root as the parent directory (/app).
WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
