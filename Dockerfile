# Multi-stage build để giảm image size
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt /app/

# Cài PyTorch riêng biệt với cache mount và index-url chính xác
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Cài các dependencies khác (loại trừ PyTorch đã cài)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    $(grep -v "torch" requirements.txt | grep -v "^--extra-index-url")

# Cài RunPod và MinIO
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runpod>=1.6.0 minio>=7.0.0

# Download models trong build stage
RUN python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('checkpoints/whisper', exist_ok=True)
hf_hub_download(repo_id='ByteDance/LatentSync-1.5', filename='whisper/tiny.pt', local_dir='checkpoints')
hf_hub_download(repo_id='ByteDance/LatentSync-1.5', filename='latentsync_unet.pt', local_dir='checkpoints')
"

# Copy source code cuối cùng
COPY . /app/

# Create directories
RUN mkdir -p /app/temp /app/output

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "rp_handler.py"]
