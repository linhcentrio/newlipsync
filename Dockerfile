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

# Install system dependencies including python3.10-devel
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
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

# Verify Python development headers are available
RUN echo "=== Verifying Python Development Environment ===" && \
    python3.10-config --cflags && \
    python3.10-config --ldflags && \
    ls -la /usr/include/python3.10/ && \
    echo "✅ Python3.10 development headers verified"

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

# Verify downloaded file với Python verification
RUN echo "=== Verifying downloaded wheel ===" && \
    ls -la /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl && \
    python -c "
import os
import zipfile
import hashlib

wheel_path = '/tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl'

# Check file exists và size
if not os.path.exists(wheel_path):
    print('❌ Wheel file not found')
    exit(1)

size = os.path.getsize(wheel_path)
print(f'📦 Wheel size: {size:,} bytes')

# Minimum expected size check (should be > 500KB)
if size < 500000:
    print('❌ Wheel file too small, likely corrupted')
    exit(1)

# Verify it's a valid ZIP file
try:
    with zipfile.ZipFile(wheel_path, 'r') as z:
        files = z.namelist()
        print(f'📋 Wheel contains {len(files)} files')
        
        # Check for essential wheel metadata
        has_metadata = any('METADATA' in f for f in files)
        has_wheel = any('WHEEL' in f for f in files)
        
        if has_metadata and has_wheel:
            print('✅ Wheel structure is valid')
        else:
            print('❌ Invalid wheel structure')
            exit(1)
            
        # Show some key files
        key_files = [f for f in files if any(x in f for x in ['insightface', '__init__', 'METADATA'])]
        print(f'🔍 Key files found: {len(key_files)}')
        
except zipfile.BadZipFile:
    print('❌ Wheel is not a valid ZIP file')
    exit(1)
except Exception as e:
    print(f'❌ Error checking wheel: {e}')
    exit(1)

print('✅ Wheel verification successful')
"

# Install InsightFace wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "=== Installing InsightFace ===" && \
    pip install /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl --force-reinstall

# Clean up wheel file
RUN rm -f /tmp/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Verify InsightFace installation với comprehensive check
RUN python -c "
import sys
print('=== InsightFace Installation Verification ===')

try:
    import insightface
    print(f'✅ InsightFace imported successfully')
    print(f'📝 Version: {insightface.__version__}')
    
    # Check if we can create a basic app instance
    try:
        app = insightface.app.FaceAnalysis()
        print('✅ FaceAnalysis app created successfully')
    except Exception as e:
        print(f'⚠️  Warning: Could not create FaceAnalysis app: {e}')
        print('   This may be normal if models are not downloaded yet')
    
    print('✅ InsightFace basic verification completed')
    
except ImportError as e:
    print(f'❌ Failed to import InsightFace: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error during InsightFace verification: {e}')
    sys.exit(1)
"

# Copy download models script và thực hiện download
COPY download_models.py /app/
RUN python download_models.py

# Verify all core dependencies
RUN python -c "
import sys
print('=== Final Dependencies Verification ===')

dependencies = {
    'torch': 'PyTorch',
    'cv2': 'OpenCV',
    'numpy': 'NumPy', 
    'transformers': 'Transformers',
    'diffusers': 'Diffusers',
    'insightface': 'InsightFace'
}

failed = []
for module, name in dependencies.items():
    try:
        exec(f'import {module}')
        print(f'✅ {name}: OK')
    except ImportError as e:
        print(f'❌ {name}: FAILED - {e}')
        failed.append(name)

if failed:
    print(f'❌ Failed dependencies: {failed}')
    sys.exit(1)

# CUDA check
import torch
print(f'🔧 PyTorch version: {torch.__version__}')
print(f'🎮 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 CUDA devices: {torch.cuda.device_count()}')
    print(f'🎮 Current device: {torch.cuda.current_device()}')

print('✅ All core dependencies verified successfully')
"

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
