# Sử dụng spxiong/pytorch image với PyTorch 2.5.1, Python 3.10.15, CUDA 12.1.0
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Verify environment trước khi bắt đầu
RUN python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')" && \
    cat /etc/os-release

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip và install tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install dependencies
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

# Install InsightFace dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    scikit-image>=0.14.2 \
    Pillow \
    matplotlib \
    scipy \
    easydict \
    cython

# Download InsightFace wheel với improved reliability
RUN echo "=== Downloading InsightFace wheel ===" && \
    (wget --no-check-certificate --timeout=30 --tries=5 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl || \
    curl -L -o /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl") && \
    echo "Download completed successfully"

# Verify downloaded file
RUN echo "=== Verifying downloaded wheel ===" && \
    ls -la /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    file /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Install InsightFace wheel - FIXED syntax
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing InsightFace ===" && \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl --force-reinstall

# Clean up wheel file
RUN rm -f /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Verify InsightFace installation với basic check
RUN python -c "import insightface; print(f'✅ InsightFace: {insightface.__version__}')" && \
    echo "✅ InsightFace basic import successful"

# Copy download models script và thực hiện download
COPY download_models.py /app/
RUN python download_models.py

# Verify all core dependencies
RUN python -c "import torch, cv2, numpy as np, transformers, diffusers; print('=== Dependencies OK ==='); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print('All core dependencies verified')"

# Copy source code
COPY . /app/

# Create working directories
RUN mkdir -p /app/temp /app/output /app/checkpoints

# Set environment variables
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/checkpoints"
ENV HF_HOME="/app/checkpoints"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "rp_handler.py"]
