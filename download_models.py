#!/usr/bin/env python3
"""
Download LatentSync models script
Separated to avoid Docker FROM parsing conflicts
"""

import os
from huggingface_hub import hf_hub_download

def main():
    """Download required model files"""
    
    # Create directories
    os.makedirs('checkpoints/whisper', exist_ok=True)
    
    print("Downloading Whisper model...")
    hf_hub_download(
        repo_id='ByteDance/LatentSync-1.5', 
        filename='whisper/tiny.pt', 
        local_dir='checkpoints'
    )
    
    print("Downloading LatentSync UNet...")
    hf_hub_download(
        repo_id='ByteDance/LatentSync-1.5', 
        filename='latentsync_unet.pt', 
        local_dir='checkpoints'
    )
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main()
