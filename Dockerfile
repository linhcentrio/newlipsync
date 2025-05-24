# Multi-stage build để giảm image size
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS base

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

# Cài PyTorch riêng biệt với cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Download và cài InsightFace prebuilt wheel
RUN wget https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl \
    -O /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Cài các dependencies khác (loại trừ insightface)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    diffusers==0.32.2 \
    transformers==4.48.0 \
    decord==0.6.0 \
    accelerate==0.26.1 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    opencv-python==4.9.0.80 \
    mediapipe==0.10.11 \
    python_speech_features==0.6 \
    librosa==0.10.1 \
    scenedetect==0.6.1 \
    ffmpeg-python==0.2.0 \
    imageio==2.31.1 \
    imageio-ffmpeg==0.5.1 \
    lpips==0.1.4 \
    face-alignment==1.4.1 \
    gradio==5.24.0 \
    huggingface-hub==0.30.2 \
    numpy==1.26.4 \
    kornia==0.8.0 \
    onnxruntime-gpu==1.21.0 \
    runpod>=1.6.0 \
    minio>=7.0.0

# Clean up wheel file
RUN rm /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Download models - Sử dụng separate script
COPY download_models.py /app/
RUN python download_models.py

# Copy source code cuối cùng
COPY . /app/

# Create directories
RUN mkdir -p /app/temp /app/output

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "rp_handler.py"]
