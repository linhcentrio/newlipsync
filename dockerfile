# Sử dụng image CUDA của NVIDIA với Ubuntu 22.04
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

###############################################################################
# 1. Cài đặt các gói hệ thống cần thiết và Python
###############################################################################
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    python3.10 \
    python3.10-venv \
    python3-pip \
    wget \
    curl \
 && rm -rf /var/lib/apt/lists/*

###############################################################################
# 2. Thiết lập thư mục làm việc
###############################################################################
WORKDIR /home/new_lipsync

###############################################################################
# 3. Copy toàn bộ mã nguồn
###############################################################################
COPY . .

###############################################################################
# 4. Tạo môi trường ảo và cài đặt các dependencies
###############################################################################
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod huggingface_hub

###############################################################################
# 5. Tải checkpoints và thiết lập các thư mục cần thiết
###############################################################################
RUN . venv/bin/activate && \
    huggingface-cli download ByteDance/LatentSync --local-dir checkpoints --exclude "*.git*" "README.md" && \
    mkdir -p /root/.cache/torch/hub/checkpoints && \
    ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip && \
    ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth && \
    ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

###############################################################################
# 6. Tạo các thư mục cần thiết và tải models
###############################################################################
RUN mkdir -p outputs \
    && mkdir -p /home/new_lipsync/enhancers/GFPGAN \
    && mkdir -p /home/new_lipsync/faceID

# Tải GFPGAN model
RUN wget -O /home/new_lipsync/enhancers/GFPGAN/gfpgan_1.4.onnx \
    https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx

# Tải FaceID recognition model
RUN wget -O /home/new_lipsync/faceID/recognition.onnx \
    https://raw.githubusercontent.com/jahongir7174/FaceID/master/weights/recognition.onnx

###############################################################################
# 7. Thiết lập biến môi trường
###############################################################################
ENV PATH="/home/new_lipsync/venv/bin:$PATH"

###############################################################################
# 8. Tạo file handler cho RunPod
###############################################################################
COPY <<'EOF' /home/new_lipsync/rp_handler.py
import runpod
import subprocess
import os

def handler(event):
    try:
        # Lấy input parameters từ request
        input_data = event["input"]
        video_path = input_data.get("video_path", "assets/demo1_video.mp4")
        audio_path = input_data.get("audio_path", "assets/demo1_audio.wav")
        output_name = input_data.get("output_name", "video_out.mp4")
        
        # Đường dẫn output
        video_out_path = f"outputs/{output_name}"
        
        # Tạo command để chạy app_cli.py
        command = [
            "/home/new_lipsync/venv/bin/python",
            "app_cli.py",
            "--inference_ckpt_path", "checkpoints/unet.pt",
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", video_out_path
        ]
        
        # Chạy command
        process = subprocess.run(
            command,
            cwd="/home/new_lipsync",
            check=True,
            capture_output=True,
            text=True
        )
        
        # Kiểm tra file output có tồn tại
        if os.path.exists(video_out_path):
            return {
                "output": {
                    "status": "success",
                    "video_path": video_out_path,
                    "message": "Processing completed successfully"
                }
            }
        else:
            return {
                "output": {
                    "status": "error",
                    "message": "Output file was not generated",
                    "logs": process.stdout
                }
            }
            
    except Exception as e:
        return {
            "output": {
                "status": "error",
                "message": str(e)
            }
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
EOF

###############################################################################
# 9. Chạy handler
###############################################################################
CMD ["python", "-u", "rp_handler.py"]