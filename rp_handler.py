#!/usr/bin/env python3
"""
RunPod Serverless Handler for LatentSync AI - Simplified Version
Sử dụng subprocess để gọi inference script và MinIO để upload
"""

import runpod
import os
import tempfile
import uuid
import requests
import subprocess
import time
from pathlib import Path
from minio import Minio
from urllib.parse import quote
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "aiclip-dfl"
MINIO_BUCKET = "aiclipdfl"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

def download_file(url: str, local_path: str) -> bool:
    """Download file từ URL"""
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded: {local_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file lên MinIO và trả về URL"""
    try:
        # Upload file
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        # Generate URL
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"Uploaded: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise e

def run_inference(video_path: str, audio_path: str, output_path: str, 
                 inference_steps: int = 20, guidance_scale: float = 2.0) -> bool:
    """Chạy LatentSync inference bằng subprocess"""
    try:
        # Construct command
        cmd = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", output_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=1000  # 5 minutes timeout
        )
        
        # Log output
        if result.stdout:
            logger.info(f"Inference stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Inference stderr: {result.stderr}")
        
        # Check if successful
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info("Inference completed successfully")
            return True
        else:
            logger.error(f"Inference failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Inference timeout")
        return False
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return False

def handler(job):
    """
    Main RunPod handler
    
    Expected input:
    {
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.wav",
        "inference_steps": 20,
        "guidance_scale": 2.0
    }
    """
    
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        # Get input
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        
        if not video_url or not audio_url:
            return {"error": "Missing video_url or audio_url"}
        
        # Optional parameters
        inference_steps = job_input.get("inference_steps", 20)
        guidance_scale = job_input.get("guidance_scale", 2.0)
        
        logger.info(f"Job {job_id}: Processing video={video_url}, audio={audio_url}")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav") 
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # Download files
            logger.info("Downloading input files...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio"}
            
            # Run inference
            logger.info("Running LatentSync inference...")
            if not run_inference(video_path, audio_path, output_path, 
                               inference_steps, guidance_scale):
                return {"error": "Inference failed"}
            
            # Check output exists
            if not os.path.exists(output_path):
                return {"error": "Output video not generated"}
            
            # Upload to MinIO
            logger.info("Uploading result...")
            output_filename = f"lipsync_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return result
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "status": "completed"
            }
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    logger.info("Starting LatentSync RunPod Serverless Worker...")
    
    # Verify environment
    try:
        import torch
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
    except:
        logger.warning("PyTorch not available")
    
    # Start RunPod
    runpod.serverless.start({"handler": handler})
