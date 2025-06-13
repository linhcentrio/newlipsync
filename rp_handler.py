#!/usr/bin/env python3
"""
RunPod Serverless Handler for LatentSync-1.6 AI with GFPGAN Enhancement
Integrated complete face enhancement logic from app_enhancher.py
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from omegaconf import OmegaConf
from tqdm import tqdm
import onnxruntime
import logging
import gc
import argparse
import subprocess
from datetime import datetime

# Thêm path để import các module local
sys.path.append('/app')

# Import các module cần thiết
try:
    from scripts.inference import main as inference_main
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    from faceID.faceID import FaceRecognition
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths for LatentSync-1.6 với stage2_512.yaml
CONFIG_PATH = Path("/app/configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("/app/checkpoints/latentsync_unet.pt")

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "aiclipdfl"
MINIO_BUCKET = "aiclipdfl"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Global model instances
detector = None
enhancer = None
recognition = None
config = None

def initialize_models():
    """Khởi tạo các model cần thiết theo đúng app_enhancher.py"""
    global detector, enhancer, recognition, config
    
    try:
        # Load LatentSync-1.6 config với stage2_512.yaml
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        config = OmegaConf.load(CONFIG_PATH)
        logger.info(f"✅ LatentSync-1.6 config loaded from {CONFIG_PATH}")
        
        # Log config info
        if hasattr(config, 'data') and hasattr(config.data, 'resolution'):
            logger.info(f"📐 Resolution: {config.data.resolution}")
        
        # Khởi tạo face detector (RetinaFace) - theo app_enhancher.py
        detector_path = "/app/utils/scrfd_2.5g_bnkps.onnx"
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"RetinaFace model not found: {detector_path}")
            
        detector = RetinaFace(
            detector_path,
            provider=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
            ],
            session_options=None
        )
        logger.info("✅ Face detector (RetinaFace) initialized")
        
        # Khởi tạo face recognition - theo app_enhancher.py
        recognition_path = "/app/faceID/recognition.onnx"
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"FaceRecognition model not found: {recognition_path}")
            
        recognition = FaceRecognition(recognition_path)
        logger.info("✅ Face recognition initialized")
        
        # Khởi tạo GFPGAN enhancer - theo app_enhancher.py logic
        enhancer_path = "/app/enhancers/GFPGAN/GFPGANv1.4.onnx"
        if not os.path.exists(enhancer_path):
            raise FileNotFoundError(f"GFPGAN model not found: {enhancer_path}")
            
        # Detect device như trong app_enhancher.py
        device = 'cpu'
        if onnxruntime.get_device() == 'GPU':
            device = 'cuda'
        logger.info(f"Running on {device}")
        
        enhancer = GFPGAN(model_path=enhancer_path, device=device)
        logger.info(f"✅ GFPGAN enhancer initialized on {device}")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise e

def download_file(url: str, local_path: str) -> bool:
    """Download file từ URL với progress tracking"""
    try:
        logger.info(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"✅ Downloaded: {local_path} ({downloaded/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file lên MinIO và trả về URL"""
    try:
        # Upload file
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        # Generate URL
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded: {file_url}")
        
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def create_args(video_path: str, audio_path: str, output_path: str, 
               inference_steps: int, guidance_scale: float, seed: int) -> argparse.Namespace:
    """Tạo arguments cho LatentSync-1.6 inference với stage2_512.yaml"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, required=True)
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    
    return parser.parse_args([
        "--unet_config_path", str(CONFIG_PATH.absolute()),
        "--inference_ckpt_path", str(CHECKPOINT_PATH.absolute()),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
    ])

def run_lipsync_inference(video_path: str, audio_path: str, output_path: str,
                         inference_steps: int = 20, guidance_scale: float = 1.5, 
                         seed: int = 1247) -> bool:
    """Chạy LatentSync-1.6 inference với stage2_512.yaml"""
    global config
    
    try:
        logger.info("🎯 Running LatentSync-1.6 inference with stage2_512.yaml...")
        
        # Update config with runtime parameters
        config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })
        
        # Create arguments
        args = create_args(video_path, audio_path, output_path, 
                          inference_steps, guidance_scale, seed)
        
        # Run inference using the main function from scripts.inference
        result = inference_main(config=config, args=args)
        
        # Check if output file exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"✅ LatentSync-1.6 inference completed successfully ({file_size:.1f} MB)")
            return True
        else:
            logger.error("❌ LatentSync-1.6 inference failed - no output file")
            return False
            
    except Exception as e:
        logger.error(f"❌ LatentSync-1.6 inference error: {e}")
        return False

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    """Xử lý batch frames - hoàn toàn theo app_enhancher.py"""
    frames, aligned_faces, mats = zip(*frame_buffer)
    enhanced_faces = enhancer.enhance_batch(aligned_faces)
    
    for frame, aligned_face, mat, enhanced_face in zip(frames, aligned_faces, mats, enhanced_faces):
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))
        blended_face = (face_mask_resized * enhanced_face_resized + (1 - face_mask_resized) * aligned_face).astype(np.uint8)
        
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))
        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)
        
        out.write(final_frame)

def enhance_video_with_gfpgan(input_video_path: str, output_path: str = None) -> bool:
    """Nâng cấp video sử dụng GFPGAN - hoàn toàn theo app_enhancher.py"""
    global detector, enhancer
    
    try:
        logger.info(f"✨ Starting GFPGAN enhancement: {input_video_path}")
        
        # Open video
        video_stream = cv2.VideoCapture(input_video_path)
        if not video_stream.isOpened():
            raise ValueError(f"Failed to open video file: {input_video_path}")
        
        # Get video properties
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path is None:
            output_path = os.path.splitext(input_video_path)[0] + '_enhanced_gfpgan.mp4'
        
        # Create temporary video path như app_enhancher.py
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        
        # Create video writer
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Create face mask - y hệt app_enhancher.py
        face_mask = np.zeros((256, 256), dtype=np.uint8)
        face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_mask = face_mask / 255
        
        # Batch processing - theo app_enhancher.py
        batch_size = 1  # Như trong app_enhancher.py
        frame_buffer = []
        
        logger.info(f"Processing {total_frames} frames with batch size {batch_size}")
        
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = video_stream.read()
            if not ret:
                break
            
            bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
            if len(kpss) == 0:
                out.write(frame)
                continue
            
            aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
            frame_buffer.append((frame, aligned_face, mat))
            
            if len(frame_buffer) >= batch_size:
                process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                frame_buffer = []
        
        # Process remaining frames
        if frame_buffer:
            process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
        
        # Cleanup video streams
        video_stream.release()
        out.release()
        
        logger.info(f"Enhanced video frames saved to {temp_video_path}")
        
        # Extract and combine audio - y hệt app_enhancher.py
        audio_path = os.path.splitext(output_path)[0] + '.aac'
        
        subprocess.run([
            'ffmpeg', '-y', '-i', input_video_path, 
            '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path
        ], check=True)
        
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, 
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', 
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path
        ], check=True)
        
        # Cleanup - y hệt app_enhancher.py
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"✅ Enhanced video with original audio saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GFPGAN enhancement failed: {e}")
        return False
    finally:
        # Cleanup
        if 'video_stream' in locals():
            video_stream.release()
        if 'out' in locals():
            out.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def handler(job):
    """
    Main RunPod handler for LatentSync-1.6 with GFPGAN
    Complete integration following app_enhancher.py logic
    
    Expected input:
    {
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.wav",
        "inference_steps": 20,
        "guidance_scale": 1.5,
        "seed": 1247,
        "enable_enhancement": true
    }
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        # Get input parameters
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        
        if not video_url or not audio_url:
            return {"error": "Missing video_url or audio_url"}
        
        # Parameters for LatentSync-1.6 với 512x512
        inference_steps = job_input.get("inference_steps", 20)
        guidance_scale = job_input.get("guidance_scale", 1.5)  # Default for v1.6
        seed = job_input.get("seed", 1247)
        enable_enhancement = job_input.get("enable_enhancement", True)
        
        logger.info(f"🚀 Job {job_id}: LatentSync-1.6 + GFPGAN Processing (512x512)")
        logger.info(f"📺 Video: {video_url}")
        logger.info(f"🎵 Audio: {audio_url}")
        logger.info(f"⚙️ Parameters: steps={inference_steps}, scale={guidance_scale}, seed={seed}, enhance={enable_enhancement}")
        logger.info(f"📐 Config: {CONFIG_PATH}")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            
            # Create output path with timestamp like gradio_app.py
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            lipsync_output_path = os.path.join(temp_dir, f"lipsync_512_{current_time}.mp4")
            final_output_path = os.path.join(temp_dir, f"final_512_{current_time}.mp4")
            
            # Step 1: Download files
            logger.info("📥 Step 1/4: Downloading input files...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio"}
            
            # Step 2: Run LatentSync-1.6 inference
            logger.info("🎯 Step 2/4: Running LatentSync-1.6 inference (512x512)...")
            if not run_lipsync_inference(video_path, audio_path, lipsync_output_path, 
                                       inference_steps, guidance_scale, seed):
                return {"error": "LatentSync-1.6 inference failed"}
            
            # Check lipsync output exists
            if not os.path.exists(lipsync_output_path):
                return {"error": "Lipsync output video not generated"}
            
            # Step 3: Face enhancement (optional)
            if enable_enhancement:
                logger.info("✨ Step 3/4: Enhancing faces with GFPGAN (following app_enhancher.py)...")
                if not enhance_video_with_gfpgan(lipsync_output_path, final_output_path):
                    logger.warning("⚠️ Face enhancement failed, using lipsync result")
                    final_output_path = lipsync_output_path
            else:
                logger.info("⏭️ Step 3/4: Skipping face enhancement")
                final_output_path = lipsync_output_path
            
            # Step 4: Upload result
            logger.info("📤 Step 4/4: Uploading result...")
            output_filename = f"latentsync16_gfpgan_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(final_output_path, output_filename)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return result
            return {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "model_version": "LatentSync-1.6",
                "resolution": "512x512",
                "config_file": "stage2_512.yaml",
                "face_enhancement": "GFPGAN",
                "face_recognition_enabled": True,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "enhancement_enabled": enable_enhancement,
                "status": "completed"
            }
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    finally:
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🚀 Starting LatentSync-1.6 + GFPGAN RunPod Serverless Worker...")
    logger.info(f"📐 Using config: {CONFIG_PATH}")
    
    # Verify environment
    try:
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"⚠️ Environment check failed: {e}")
    
    # Initialize models
    try:
        initialize_models()
        logger.info("✅ All models initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        sys.exit(1)
    
    # Start RunPod
    logger.info("🎬 Ready to process LatentSync-1.6 + GFPGAN requests with 512x512 resolution...")
    runpod.serverless.start({"handler": handler})
