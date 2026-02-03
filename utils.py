import numpy as np
import cv2
import paddle
import sys
import os

def tensor_to_cv2_img(image):
    """
    Converts a ComfyUI image tensor (batch) to a list of OpenCV-compatible numpy arrays (BGR).
    """
    # Assuming input is a torch tensor or numpy array [B, H, W, C] or [H, W, C] in range 0-1
    if hasattr(image, 'cpu'):
        image = image.cpu().numpy()
        
    results = []
    
    # Handle single image vs batch
    if len(image.shape) == 3:
        # [H, W, C] -> list of one
        batch_images = [image]
    else:
        # [B, H, W, C]
        batch_images = image
        
    for img in batch_images:
        # Convert 0-1 float to 0-255 uint8
        img_255 = (img * 255).astype(np.uint8)
        # Convert RGB (ComfyUI default) to BGR (OpenCV default)
        img_bgr = cv2.cvtColor(img_255, cv2.COLOR_RGB2BGR)
        results.append(img_bgr)
        
    return results

def get_paddle_hw_kwargs():
    """
    Intelligently determines the hardware arguments for PaddleOCR initialization.
    Ports logic from PaddleOCR's tools/infer/utility.py to ensure best compatibility
    across Windows, Mac (MPS/CPU), and Linux (CUDA/ROCm).
    """
    kwargs = {}
    
    # 1. Detect Hardware
    # Check for CUDA/ROCm
    if paddle.is_compiled_with_cuda() or paddle.device.is_compiled_with_rocm():
        kwargs['use_gpu'] = True
        print("DEBUG: PaddleOCR using GPU (CUDA/ROCm)")
    else:
        kwargs['use_gpu'] = False
        print("DEBUG: PaddleOCR using CPU (No CUDA detected)")

    # Check for Mac MPS (Metal Performance Shaders)
    # Paddle 2.5+ supports MPS conceptually, but often via 'gpu' flag on Mac or specific backends.
    # However, standard PaddleOCR 'use_gpu' often implies NVIDIA. 
    # Current PaddleOCR logic generally uses `use_gpu=True` if available.
    if sys.platform == 'darwin':
        # On Mac, check if we can simply use CPU to be safe, or enable mkldnn
        # MKLDNN on Mac (Arm64) is often supported and faster than vanilla CPU.
        # But 'enable_mkldnn' often defaults to None/False in basic usage.
        
        # If user installed paddlepaddle-gpu (which doesn't really exist for mac m-series the same way),
        # usually it's just 'paddlepaddle' which uses cpu/accelerate.
        pass

    # 2. Handle OneDNN (MKLDNN)
    # Windows typically has issues with OneDNN + some AVX instrs or specific Paddle versions.
    # The safest bet for Windows is enable_mkldnn=False unless we test compatibility.
    # Linux usually benefits from it if CPU only.
    
    if sys.platform == 'win32':
        # KNOWN ISSUE: PaddleOCR on Windows with MKLDNN can crash (NotImplementedError).
        # We explicitly disable it to be safe.
        kwargs['enable_mkldnn'] = False
        print("DEBUG: Forced enable_mkldnn=False for Windows compatibility")
    else:
        # On Linux/Mac, we can try to leave it default (None) or True.
        # PaddleOCR default is often False/str2bool(None) -> False in CLI, 
        # but internal default might vary.
        # To match 'PaddleOCR_Node' previous fix where we didn't specify it (defaulting to logic),
        # or where we explicitly set it False for safety.
        # Given we want "Optimized", we can try `True` for CPU cases on Linux, 
        # BUT reliability is priority. Let's keep it default (don't set key) unless we are sure.
        # Actually, let's allow it to be default (Paddle handles it).
        pass

    # 3. Check for XPU / NPU (Custom Hardware)
    # If compiled with custom devices, we might want to flag them.
    if hasattr(paddle, 'is_compiled_with_custom_device'):
        try:
            if paddle.is_compiled_with_custom_device('npu'):
                kwargs['use_npu'] = True
                print("DEBUG: PaddleOCR using NPU")
            if paddle.is_compiled_with_custom_device('xpu'):
                kwargs['use_xpu'] = True
                print("DEBUG: PaddleOCR using XPU")
            if paddle.is_compiled_with_custom_device('mlu'):
                kwargs['use_mlu'] = True
                print("DEBUG: PaddleOCR using MLU")
        except Exception:
            pass
            
    return kwargs
