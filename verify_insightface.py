#!/usr/bin/env python3
"""
InsightFace verification script
Separated to avoid Docker FROM parsing conflicts
"""

def verify_insightface():
    """Verify InsightFace installation and functionality"""
    
    print("=== InsightFace Verification ===")
    
    try:
        import insightface
        print(f"✅ InsightFace imported: {insightface.__version__}")
        
        # Test app creation (might fail on CPU-only or without models)
        try:
            app = insightface.app.FaceAnalysis()
            print("✅ FaceAnalysis app created successfully")
        except Exception as e:
            print(f"⚠️ FaceAnalysis app creation failed: {e}")
            print("This is normal if no GPU or models are available")
        
        return True
        
    except ImportError as e:
        print(f"❌ InsightFace import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = verify_insightface()
    if not success:
        exit(1)
    print("✅ InsightFace verification completed")
