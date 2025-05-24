#!/usr/bin/env python3
"""
RunPod Serverless Handler for LatentSync AI
Optimized for production deployment with comprehensive error handling,
progress updates, and MinIO integration.
"""

import runpod
import os
import tempfile
import uuid
import requests
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import quote, urlparse

import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
from minio import Minio
from minio.error import S3Error

# Import LatentSync modules
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

import logging

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

# MinIO Configuration
MINIO_CONFIG = {
    "endpoint": "108.181.198.160:9000",
    "access_key": "minioadmin",
    "secret_key": "aiclip-dfl", 
    "bucket": "aiclipdfl",
    "secure": False
}

# Model Configuration
MODEL_CONFIG = {
    "config_path": "configs/unet/stage2.yaml",
    "unet_checkpoint": "checkpoints/latentsync_unet.pt",
    "whisper_tiny": "checkpoints/whisper/tiny.pt",
    "whisper_small": "checkpoints/whisper/small.pt",
    "vae_model": "stabilityai/sd-vae-ft-mse"
}

# Processing Configuration
DEFAULT_PARAMS = {
    "inference_steps": 20,
    "guidance_scale": 2.0,
    "num_frames": 16,
    "resolution": 512,
    "seed": -1
}

# ==================== UTILITY FUNCTIONS ====================

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return filename[:100]  # Limit length

def get_file_extension(url: str) -> str:
    """Extract file extension from URL"""
    path = urlparse(url).path
    return os.path.splitext(path)[1].lower()

# ==================== MINIO CLIENT ====================

class MinIOManager:
    """Enhanced MinIO client with error handling and retry logic"""
    
    def __init__(self):
        self.client = Minio(
            MINIO_CONFIG["endpoint"],
            access_key=MINIO_CONFIG["access_key"],
            secret_key=MINIO_CONFIG["secret_key"],
            secure=MINIO_CONFIG["secure"]
        )
        self.bucket = MINIO_CONFIG["bucket"]
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure bucket exists"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except Exception as e:
            logger.warning(f"Could not ensure bucket exists: {e}")
    
    def upload_file(self, local_path: str, object_name: str = None, 
                   content_type: str = "video/mp4") -> str:
        """Upload file to MinIO with retry logic"""
        if object_name is None:
            object_name = clean_filename(os.path.basename(local_path))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Upload with metadata
                self.client.fput_object(
                    self.bucket, 
                    object_name, 
                    local_path,
                    content_type=content_type
                )
                
                # Generate direct URL
                file_name_encoded = quote(object_name)
                protocol = "https" if MINIO_CONFIG["secure"] else "http"
                file_url = f"{protocol}://{MINIO_CONFIG['endpoint']}/{self.bucket}/{file_name_encoded}"
                
                logger.info(f"Successfully uploaded: {file_url}")
                return file_url
                
            except S3Error as e:
                logger.error(f"MinIO upload attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff

# ==================== LATENTSYNC PROCESSOR ====================

class LatentSyncProcessor:
    """Optimized LatentSync processor for RunPod Serverless"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = self._get_optimal_dtype()
        self.models_loaded = False
        self.config = None
        self.pipeline = None
        
        logger.info(f"Initializing LatentSync on {self.device} with {self.dtype}")
        
        # Initialize MinIO
        self.minio = MinIOManager()
        
    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal dtype based on GPU capability"""
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere and newer
                return torch.float16
            elif capability[0] >= 7:  # Volta and newer  
                return torch.float16
        return torch.float32
    
    def _load_models(self, job_id: str):
        """Load models with progress updates"""
        if self.models_loaded:
            return
            
        try:
            runpod.serverless.progress_update(job_id, "Loading configuration...")
            
            # Load config
            self.config = OmegaConf.load(MODEL_CONFIG["config_path"])
            
            runpod.serverless.progress_update(job_id, "Loading VAE...")
            
            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                MODEL_CONFIG["vae_model"], 
                torch_dtype=self.dtype
            )
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0
            
            runpod.serverless.progress_update(job_id, "Loading audio encoder...")
            
            # Load Audio Encoder
            if self.config.model.cross_attention_dim == 768:
                whisper_path = MODEL_CONFIG["whisper_small"]
            elif self.config.model.cross_attention_dim == 384:
                whisper_path = MODEL_CONFIG["whisper_tiny"]
            else:
                raise ValueError("cross_attention_dim must be 768 or 384")
            
            audio_encoder = Audio2Feature(
                model_path=whisper_path,
                device=self.device,
                num_frames=self.config.data.num_frames,
                audio_feat_length=self.config.data.audio_feat_length,
            )
            
            runpod.serverless.progress_update(job_id, "Loading UNet...")
            
            # Load UNet
            denoising_unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(self.config.model),
                MODEL_CONFIG["unet_checkpoint"],
                device="cpu",
            )
            denoising_unet = denoising_unet.to(dtype=self.dtype)
            
            runpod.serverless.progress_update(job_id, "Loading scheduler...")
            
            # Load Scheduler
            scheduler = DDIMScheduler.from_pretrained("configs")
            
            runpod.serverless.progress_update(job_id, "Assembling pipeline...")
            
            # Create Pipeline
            self.pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                denoising_unet=denoising_unet,
                scheduler=scheduler,
            ).to(self.device)
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise e
    
    def download_file(self, url: str, local_path: str, job_id: str, 
                     file_type: str = "file") -> bool:
        """Download file with progress updates"""
        try:
            runpod.serverless.progress_update(job_id, f"Downloading {file_type}...")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every 10%
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if progress % 10 < 1:  # Approximate 10% intervals
                                runpod.serverless.progress_update(
                                    job_id, 
                                    f"Downloading {file_type}: {progress:.0f}%"
                                )
            
            logger.info(f"Downloaded {file_type}: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {file_type} from {url}: {e}")
            return False
    
    def process_lipsync(self, video_path: str, audio_path: str, job_id: str,
                       inference_steps: int = 20, guidance_scale: float = 2.0,
                       seed: int = -1) -> str:
        """Process lip sync with progress updates"""
        
        # Generate output filename
        output_filename = f"lipsync_{uuid.uuid4().hex}.mp4"
        temp_output = f"/tmp/{output_filename}"
        
        try:
            runpod.serverless.progress_update(job_id, "Starting lip sync processing...")
            
            # Set seed
            if seed != -1:
                set_seed(seed)
            else:
                torch.seed()
            
            initial_seed = torch.initial_seed()
            logger.info(f"Using seed: {initial_seed}")
            
            # Run pipeline
            self.pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=temp_output,
                video_mask_path=temp_output.replace(".mp4", "_mask.mp4"),
                num_frames=self.config.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=self.dtype,
                width=self.config.data.resolution,
                height=self.config.data.resolution,
                mask_image_path=self.config.data.mask_image_path,
            )
            
            runpod.serverless.progress_update(job_id, "Lip sync processing completed!")
            
            if not os.path.exists(temp_output):
                raise RuntimeError("Output video was not generated")
            
            return temp_output
            
        except Exception as e:
            logger.error(f"Lip sync processing failed: {e}")
            raise e
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==================== GLOBAL PROCESSOR INSTANCE ====================

processor = None

def get_processor() -> LatentSyncProcessor:
    """Get or create processor instance (singleton pattern)"""
    global processor
    if processor is None:
        processor = LatentSyncProcessor()
    return processor

# ==================== MAIN HANDLER ====================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler
    
    Expected input:
    {
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.wav",
        "inference_steps": 20,      # Optional
        "guidance_scale": 2.0,      # Optional  
        "seed": 1247               # Optional
    }
    
    Returns:
    {
        "output_video_url": "https://minio-url/output.mp4",
        "processing_info": {...},
        "status": "completed"
    }
    """
    
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        # Extract and validate input
        job_input = job.get("input", {})
        
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        
        if not video_url or not audio_url:
            return {
                "error": "Missing required inputs: video_url and audio_url are required",
                "status": "failed"
            }
        
        # Validate URLs
        if not validate_url(video_url) or not validate_url(audio_url):
            return {
                "error": "Invalid URL format for video_url or audio_url",
                "status": "failed"
            }
        
        # Extract optional parameters
        inference_steps = job_input.get("inference_steps", DEFAULT_PARAMS["inference_steps"])
        guidance_scale = job_input.get("guidance_scale", DEFAULT_PARAMS["guidance_scale"])
        seed = job_input.get("seed", DEFAULT_PARAMS["seed"])
        
        # Validate parameters
        if not (1 <= inference_steps <= 100):
            return {
                "error": "inference_steps must be between 1 and 100",
                "status": "failed"
            }
        
        if not (0.1 <= guidance_scale <= 20.0):
            return {
                "error": "guidance_scale must be between 0.1 and 20.0", 
                "status": "failed"
            }
        
        logger.info(f"Processing job {job_id}: video={video_url}, audio={audio_url}")
        
        runpod.serverless.progress_update(job_id, "Initializing processor...")
        
        # Get processor and load models
        proc = get_processor()
        proc._load_models(job_id)
        
        # Create temporary directory for this job
        with tempfile.TemporaryDirectory(prefix=f"lipsync_{job_id}_") as temp_dir:
            
            # Download input files
            video_ext = get_file_extension(video_url) or ".mp4"
            audio_ext = get_file_extension(audio_url) or ".wav"
            
            video_path = os.path.join(temp_dir, f"input_video{video_ext}")
            audio_path = os.path.join(temp_dir, f"input_audio{audio_ext}")
            
            # Download video
            if not proc.download_file(video_url, video_path, job_id, "video"):
                return {
                    "error": f"Failed to download video from: {video_url}",
                    "status": "failed"
                }
            
            # Download audio  
            if not proc.download_file(audio_url, audio_path, job_id, "audio"):
                return {
                    "error": f"Failed to download audio from: {audio_url}",
                    "status": "failed"
                }
            
            # Verify downloaded files
            if not os.path.exists(video_path) or os.path.getsize(video_path
