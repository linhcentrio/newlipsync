#!/usr/bin/env python3
"""Comprehensive environment verification"""

import sys
import platform
import subprocess

def check_system():
    print("=== System Information ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
def check_pytorch():
    print("\n=== PyTorch Environment ===")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"✅ GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)")
        
        # Test tensor operation
        x = torch.randn(2, 3)
        if torch.cuda.is_available():
            x = x.cuda()
        print("✅ Tensor operations working")
        
    except Exception as e:
        print(f"❌ PyTorch error: {e}")

def check_dependencies():
    print("\n=== Dependencies Check ===")
    deps = [
        'cv2', 'numpy', 'transformers', 'diffusers', 
        'librosa', 'omegaconf', 'runpod', 'minio'
    ]
    
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: not found")

def check_insightface():
    print("\n=== InsightFace Check ===")
    try:
        import insightface
        print(f"✅ InsightFace: {insightface.__version__}")
        
        # Test app creation
        app = insightface.app.FaceAnalysis()
        print("✅ FaceAnalysis app created")
        
    except ImportError as e:
        print(f"❌ InsightFace import: {e}")
    except Exception as e:
        print(f"⚠️ InsightFace app creation: {e}")

if __name__ == "__main__":
    check_system()
    check_pytorch()
    check_dependencies()
    check_insightface()
    print("\n=== Verification Complete ===")
