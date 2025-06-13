#!/usr/bin/env python3
"""
RunPod Serverless Handler for AI Video Processing with Face Enhancement
Optimized version with selective face enhancement
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

# Add path for local modules
sys.path.append('/app')

# Import required modules
try:
    from scripts.inference import main as inference_main
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    from faceID.faceID import FaceRecognition
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_PATH = Path("/app/configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("/app/checkpoints/latentsync_unet.pt")

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "a9TFRtBi8q3Nvj5P5Ris"
MINIO_SECRET_KEY = "fCFngM7YTr6jSkBKXZ9BkfDdXrStYXm43UGa0OZQ"
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
    """Initialize all required models"""
    global detector, enhancer, recognition, config
    
    try:
        # Load configuration
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        config = OmegaConf.load(CONFIG_PATH)
        logger.info(f"✅ Configuration loaded")
        
        # Initialize face detector
        detector_path = "/app/utils/scrfd_2.5g_bnkps.onnx"
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Face detector model not found: {detector_path}")
            
        detector = RetinaFace(
            detector_path,
            provider=[
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
            ],
            session_options=None
        )
        logger.info("✅ Face detector initialized")
        
        # Initialize face recognition
        recognition_path = "/app/faceID/recognition.onnx"
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Face recognition model not found: {recognition_path}")
            
        recognition = FaceRecognition(recognition_path)
        logger.info("✅ Face recognition initialized")
        
        # Initialize face enhancer
        enhancer_path = "/app/enhancers/GFPGAN/GFPGANv1.4.onnx"
        if not os.path.exists(enhancer_path):
            raise FileNotFoundError(f"Face enhancer model not found: {enhancer_path}")
        
        device = 'cpu'
        if onnxruntime.get_device() == 'GPU':
            device = 'cuda'
        logger.info(f"Running on {device}")
        
        enhancer = GFPGAN(model_path=enhancer_path, device=device)
        logger.info(f"✅ Face enhancer initialized on {device}")

    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise e

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
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
    """Upload file to MinIO with enhanced error handling"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded successfully: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def create_args(video_path: str, audio_path: str, output_path: str, 
               inference_steps: int, guidance_scale: float, seed: int) -> argparse.Namespace:
    """Create arguments for AI inference"""
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
    """Run AI lipsync inference"""
    global config
    
    try:
        logger.info("🎯 Running AI lipsync inference...")
        
        config["run"].update({
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        })
        
        args = create_args(video_path, audio_path, output_path, 
                          inference_steps, guidance_scale, seed)
        
        result = inference_main(config=config, args=args)
        
        if os.path.exists(output_path):
            logger.info(f"✅ Lipsync inference completed successfully")
            return True
        else:
            logger.error("❌ Lipsync inference failed - no output file")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lipsync inference error: {e}")
        return False

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    """Process batch of frames with face enhancement"""
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

def enhance_video_with_gfpgan(input_video_path: str, output_path: str = None) -> tuple[bool, dict]:
    """
    Apply selective face enhancement to video
    - Frames with faces: enhanced
    - Frames without faces: kept original
    """
    global detector, enhancer
    
    # Statistics tracking
    stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "frames_without_faces": 0,
        "faces_enhanced": 0,
        "enhancement_applied": False
    }
    
    try:
        logger.info(f"✨ Starting selective face enhancement: {input_video_path}")
        
        # Open video
        video_stream = cv2.VideoCapture(input_video_path)
        if not video_stream.isOpened():
            raise ValueError(f"Failed to open video file: {input_video_path}")
        
        # Get video properties
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stats["total_frames"] = total_frames
        
        if output_path is None:
            output_path = os.path.splitext(input_video_path)[0] + '_enhanced_gfpgan.mp4'
        
        # Create temporary video path
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        
        # Create video writer
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Create face mask for blending
        face_mask = np.zeros((256, 256), dtype=np.uint8)
        face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_mask = face_mask / 255
        
        # Batch processing
        batch_size = 1
        frame_buffer = []
        
        logger.info(f"Processing {total_frames} frames with selective enhancement...")
        
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = video_stream.read()
            if not ret:
                break
            
            # Detect faces in current frame
            bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
            
            if len(kpss) == 0:
                # No face → Keep original frame
                stats["frames_without_faces"] += 1
                out.write(frame)
                continue
            
            # Face detected → Enhance with AI
            stats["frames_with_faces"] += 1
            
            # Get aligned face
            aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
            frame_buffer.append((frame, aligned_face, mat))
            
            # Process batch when full
            if len(frame_buffer) >= batch_size:
                process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                stats["faces_enhanced"] += len(frame_buffer)
                frame_buffer = []
        
        # Process remaining frames
        if frame_buffer:
            process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
            stats["faces_enhanced"] += len(frame_buffer)
        
        # Cleanup video streams
        video_stream.release()
        out.release()
        
        # Update statistics
        if stats["frames_with_faces"] > 0:
            stats["enhancement_applied"] = True
            logger.info(f"✅ Face enhancement applied to {stats['frames_with_faces']}/{stats['total_frames']} frames")
        else:
            logger.info(f"ℹ️ No faces detected. All frames kept original.")
            stats["enhancement_applied"] = False
        
        # Extract and combine audio
        audio_path = os.path.splitext(output_path)[0] + '.aac'
        
        # Extract audio from original video
        subprocess.run([
            'ffmpeg', '-y', '-i', input_video_path, 
            '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path
        ], check=True, capture_output=True)
        
        # Combine enhanced video with audio
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, 
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', 
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path
        ], check=True, capture_output=True)
        
        # Cleanup temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"✅ Video processing completed: {output_path}")
        return True, stats
        
    except Exception as e:
        logger.error(f"❌ Face enhancement failed: {e}")
        return False, stats
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
    Main RunPod handler for AI video processing
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
        
        # Processing parameters
        inference_steps = job_input.get("inference_steps", 20)
        guidance_scale = job_input.get("guidance_scale", 1.5)
        seed = job_input.get("seed", 1247)
        enable_enhancement = job_input.get("enable_enhancement", True)
        
        logger.info(f"🚀 Job {job_id}: AI Video Processing")
        logger.info(f"📺 Video: {video_url}")
        logger.info(f"🎵 Audio: {audio_url}")
        logger.info(f"⚙️ Parameters: steps={inference_steps}, scale={guidance_scale}, seed={seed}, enhance={enable_enhancement}")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # File paths
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            lipsync_output_path = os.path.join(temp_dir, f"lipsync_{current_time}.mp4")
            final_output_path = os.path.join(temp_dir, f"final_{current_time}.mp4")
            
            # Step 1: Download input files
            logger.info("📥 Step 1/4: Downloading input files...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio"}
            
            # Step 2: Run AI lipsync
            logger.info("🎯 Step 2/4: Running AI lipsync processing...")
            if not run_lipsync_inference(video_path, audio_path, lipsync_output_path, 
                                       inference_steps, guidance_scale, seed):
                return {"error": "AI lipsync processing failed"}
            
            # Check lipsync output exists
            if not os.path.exists(lipsync_output_path):
                return {"error": "Lipsync output not generated"}
            
            # Step 3: Apply face enhancement if enabled
            enhancement_success = False
            face_enhancement_status = "disabled"
            
            if enable_enhancement:
                logger.info("✨ Step 3/4: Applying face enhancement...")
                enhancement_success, face_stats = enhance_video_with_gfpgan(lipsync_output_path, final_output_path)
                
                if not enhancement_success:
                    logger.warning("⚠️ Face enhancement failed, using lipsync result")
                    final_output_path = lipsync_output_path
                    face_enhancement_status = "failed"
                elif face_stats["enhancement_applied"]:
                    face_enhancement_status = "applied"
                    logger.info(f"✅ Enhanced {face_stats['frames_with_faces']}/{face_stats['total_frames']} frames")
                else:
                    face_enhancement_status = "no_faces_detected"
                    logger.info("ℹ️ No faces detected, keeping original quality")
            else:
                logger.info("⏭️ Step 3/4: Face enhancement disabled")
                final_output_path = lipsync_output_path
            
            # Ensure final output exists
            if not os.path.exists(final_output_path):
                if os.path.exists(lipsync_output_path):
                    import shutil
                    shutil.copy2(lipsync_output_path, final_output_path)
                else:
                    return {"error": "No output file generated"}
            
            # Step 4: Upload result
            logger.info("📤 Step 4/4: Uploading result...")
            output_filename = f"ai_video_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(final_output_path, output_filename)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare simplified response (security-focused)
            response = {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "face_enhancement": face_enhancement_status,
                "status": "completed"
            }
            
            # Add face stats only if enhancement was attempted
            if enable_enhancement and face_stats["total_frames"] > 0:
                response["face_enhancement_stats"] = {
                    "total_frames": face_stats["total_frames"],
                    "frames_enhanced": face_stats["frames_with_faces"],
                    "enhancement_rate": round(face_stats["frames_with_faces"] / face_stats["total_frames"] * 100, 1)
                }
            
            return response
            
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
    logger.info("🚀 Starting AI Video Processing Serverless Worker...")
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    
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
    
    # Start RunPod serverless worker
    logger.info("🎬 Ready to process AI video requests...")
    runpod.serverless.start({"handler": handler})
