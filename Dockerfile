# Sử dụng spxiong/pytorch image với PyTorch 2.5.1, Python 3.10.15, CUDA 12.1.0
FROM spxiong/pytorch:2.5.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    libgl1 \
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

# Install main dependencies từ requirements.txt (trừ InsightFace)
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
    DeepCache==0.1.1

# Install additional dependencies for RunPod
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    runpod>=1.6.0 \
    minio>=7.0.0

# Install InsightFace dependencies (cần thiết cho InsightFace)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    scikit-image>=0.14.2 \
    Pillow \
    matplotlib \
    scipy \
    easydict \
    cython

# Install InsightFace (có thể cần fallback strategy)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Attempting to install InsightFace via pip ===" && \
    pip install insightface==0.7.3 --no-cache-dir || \
    (echo "=== Pip install failed, downloading wheel ===" && \
    wget --no-check-certificate --timeout=30 --tries=3 \
    "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl" \
    -O /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl --force-reinstall && \
    rm -f /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl)

# Verify installations
RUN echo "=== Verifying key packages ===" && \
    python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" && \
    python -c "import diffusers; print(f'✅ Diffusers: {diffusers.__version__}')" && \
    python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" && \
    python -c "import DeepCache; print(f'✅ DeepCache: imported successfully')" && \
    python -c "import insightface; print(f'✅ InsightFace: {insightface.__version__}')" && \
    python -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')" && \
    python -c "import onnxruntime; print(f'✅ ONNXRuntime: {onnxruntime.__version__}')" && \
    python -c "import runpod; print(f'✅ RunPod: imported successfully')" && \
    python -c "import minio; print(f'✅ MinIO: imported successfully')"

# Copy source code FIRST (để có đầy đủ cấu trúc thư mục)
COPY . /app/

# Download LatentSync-1.6 models
RUN echo "=== Downloading LatentSync-1.6 models ===" && \
    mkdir -p /app/checkpoints/whisper && \
    huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints && \
    echo "✅ LatentSync models downloaded"

# Download GFPGAN model - trực tiếp vào đúng vị trí
RUN echo "=== Downloading GFPGAN model ===" && \
    mkdir -p /app/enhancers/GFPGAN && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GFPGANv1.4.onnx" \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    echo "✅ GFPGAN model downloaded"

# Download RetinaFace model - trực tiếp vào đúng vị trí
RUN echo "=== Downloading RetinaFace model ===" && \
    mkdir -p /app/utils && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/utils/scrfd_2.5g_bnkps.onnx && \
    echo "✅ RetinaFace model downloaded"

# Download FaceRecognition model - trực tiếp vào đúng vị trí
RUN echo "=== Downloading FaceRecognition model ===" && \
    mkdir -p /app/faceID && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "http://108.181.198.160:9000/aiclipdfl/recognition.onnx" \
    -O /app/faceID/recognition.onnx && \
    echo "✅ FaceRecognition model downloaded"

# Verify tất cả model files exist
RUN echo "=== Verifying model files ===" && \
    ls -la /app/configs/unet/ && \
    test -f /app/configs/unet/stage2_512.yaml && echo "✅ Config stage2_512.yaml verified" && \
    test -f /app/enhancers/GFPGAN/GFPGANv1.4.onnx && echo "✅ GFPGAN model verified" && \
    test -f /app/utils/scrfd_2.5g_bnkps.onnx && echo "✅ RetinaFace model verified" && \
    test -f /app/faceID/recognition.onnx && echo "✅ FaceRecognition model verified" && \
    test -f /app/checkpoints/latentsync_unet.pt && echo "✅ LatentSync UNet verified" && \
    test -f /app/checkpoints/whisper/tiny.pt && echo "✅ Whisper model verified"

# Show file sizes for verification
RUN echo "=== Model file sizes ===" && \
    ls -lh /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    ls -lh /app/utils/scrfd_2.5g_bnkps.onnx && \
    ls -lh /app/faceID/recognition.onnx && \
    ls -lh /app/checkpoints/latentsync_unet.pt && \
    ls -lh /app/checkpoints/whisper/tiny.pt

# Final import test để đảm bảo tất cả dependencies có thể import
RUN echo "=== Final import test ===" && \
    python -c "
import sys
print(f'Python: {sys.version}')

# Test all major imports
import torch
print(f'PyTorch: {torch.__version__}')

import diffusers
from diffusers import DDIMScheduler, AutoencoderKL
print(f'Diffusers: {diffusers.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

import DeepCache
print('DeepCache: OK')

import cv2
print(f'OpenCV: {cv2.__version__}')

import numpy as np
print(f'NumPy: {np.__version__}')

import onnxruntime
print(f'ONNXRuntime: {onnxruntime.__version__}')

import insightface
print(f'InsightFace: {insightface.__version__}')

import runpod
print('RunPod: OK')

import minio
print('MinIO: OK')

import omegaconf
print('OmegaConf: OK')

import librosa
print('Librosa: OK')

import face_alignment
print('Face Alignment: OK')

import tqdm
print('TQDM: OK')

print('🎉 All dependencies imported successfully!')
"

# Set environment variables
ENV PYTHONPATH="/app"
ENV TORCH_HOME="/app/checkpoints"
ENV HF_HOME="/app/checkpoints"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "rp_handler.py"]
