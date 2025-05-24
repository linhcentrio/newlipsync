# RECOMMENDED: Sử dụng xiongsp fork thay vì cnstark
FROM spxiong/pytorch:2.4.1-py3.10.15-cuda12.1.0-ubuntu22.04 AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Verify environment
RUN python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install packages
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

# Install InsightFace
RUN wget https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl \
    -O /tmp/insightface.whl && \
    pip install /tmp/insightface.whl --force-reinstall --no-deps && \
    rm /tmp/insightface.whl

# Download models
COPY download_models.py /app/
RUN python download_models.py

# Copy source code
COPY . /app/
RUN mkdir -p /app/temp /app/output

ENV PYTHONPATH="/app:${PYTHONPATH}"
CMD ["python", "rp_handler.py"]
