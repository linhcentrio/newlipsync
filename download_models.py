#!/usr/bin/env python3
"""
Download LatentSync-1.6 models script
Only download models, no config files (already exist in source)
"""

import os
import requests
from huggingface_hub import hf_hub_download

def download_direct(url: str, local_path: str) -> bool:
    """Download file directly from URL"""
    try:
        print(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Downloaded: {local_path} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Download required model files for LatentSync-1.6"""
    
    print("=== Downloading LatentSync-1.6 Models ===")
    
    # Create checkpoints directory
    os.makedirs('checkpoints/whisper', exist_ok=True)
    
    print("📥 Downloading Whisper model...")
    hf_hub_download(
        repo_id='ByteDance/LatentSync-1.6',
        filename='whisper/tiny.pt',
        local_dir='checkpoints'
    )
    print("✅ Whisper model downloaded")
    
    print("📥 Downloading LatentSync UNet...")
    hf_hub_download(
        repo_id='ByteDance/LatentSync-1.6',
        filename='latentsync_unet.pt',
        local_dir='checkpoints'
    )
    print("✅ LatentSync UNet downloaded")
    
    print("📥 Downloading GFPGAN model...")
    download_direct(
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GFPGANv1.4.onnx",
        "enhancers/GFPGAN/GFPGANv1.4.onnx"
    )
    
    print("📥 Downloading RetinaFace model...")
    download_direct(
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx",
        "utils/scrfd_2.5g_bnkps.onnx"
    )
    
    print("📥 Downloading FaceRecognition model...")
    download_direct(
        "http://108.181.198.160:9000/aiclipdfl/recognition.onnx",
        "faceID/recognition.onnx"
    )
    
    # Verify files exist
    required_files = [
        'checkpoints/whisper/tiny.pt',
        'checkpoints/latentsync_unet.pt',
        'enhancers/GFPGAN/GFPGANv1.4.onnx',
        'utils/scrfd_2.5g_bnkps.onnx',
        'faceID/recognition.onnx'
    ]
    
    print("\n=== Verifying Downloads ===")
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_files_exist = False
    
    # Check config file exists (should already be in source)
    if os.path.exists('configs/unet/stage2_512.yaml'):
        print(f"✅ configs/unet/stage2_512.yaml - EXISTS")
    else:
        print(f"❌ configs/unet/stage2_512.yaml - NOT FOUND")
        all_files_exist = False
    
    if all_files_exist:
        print("\n🎉 All models downloaded and verified successfully!")
    else:
        print("\n❌ Some models failed to download!")
        exit(1)

if __name__ == "__main__":
    main()
