import numpy as np
import cv2

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
