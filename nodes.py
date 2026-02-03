
import sys
import os
import traceback
from .utils import tensor_to_cv2_img, get_paddle_hw_kwargs

# Attempt to import PaddleOCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

class PaddleOCR_Node:
    """
    Main PaddleOCR Custom Node.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "language": (["ch", "en", "japan", "korean", "chinese_cht", "french", "german"], {"default": "ch"}),
                # Renamed from use_angle_cls
                "vertical_direction": ("BOOLEAN", {"default": True}),
                # Added ocr_version
                "ocr_version": (["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"], {"default": "PP-OCRv5"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "apply_ocr"
    CATEGORY = "PaddleOCR"

    def apply_ocr(self, image, language, vertical_direction, ocr_version):
        try:
            if PaddleOCR is None:
                raise ImportError("PaddleOCR library is not installed.")

            print(f"DEBUG: Initializing PaddleOCR. Lang: {language}, Vertical: {vertical_direction}, Version: {ocr_version}")

            # Instantiate PaddleOCR
            # We pass 'use_textline_orientation' (which vertical_direction maps to)
            # and 'ocr_version' to let the internal logic handle model selection.
            # Get hardware kwargs (handles GPU/CPU/OneDNN automatically)
            hw_kwargs = get_paddle_hw_kwargs()
            print(f"DEBUG: Hardware Kwargs: {hw_kwargs}")

            try:
                 ocr = PaddleOCR(
                     use_textline_orientation=vertical_direction, 
                     lang=language,
                     ocr_version=ocr_version,
                     **hw_kwargs
                 )
            except TypeError as e:
                 print(f"DEBUG: Initialization TypeError: {e}")
                 # Fallback for older/standard versions that might not support keys
                 # We try 'use_angle_cls' if 'use_textline_orientation' fails, etc.
                 # But since the user is using the Pipeline wrapper, the above SHOULD work.
                 try:
                     ocr = PaddleOCR(use_angle_cls=vertical_direction, lang=language, **hw_kwargs)
                 except:
                     ocr = PaddleOCR(lang=language, **hw_kwargs)
            
            # process
            cv_images = tensor_to_cv2_img(image)
            full_text_results = []
            
            for i, img_numpy in enumerate(cv_images):
                # ocr() method
                try:
                    result = ocr.ocr(img_numpy, use_textline_orientation=vertical_direction)
                except TypeError:
                    # Fallback
                    result = ocr.ocr(img_numpy, cls=vertical_direction)
                
                if not result:
                    continue
                
                if result[0] is None:
                    continue    

                # Flatten 
                lines = result
                # Handle batch or odd structure
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and isinstance(result[0][0], list):
                     lines = result[0]

                for line in lines:
                    # Handle if line is dictionary (PaddleX Pipeline structure)
                    if isinstance(line, dict):
                         rec_texts = line.get('rec_texts', [])
                         if isinstance(rec_texts, list):
                             full_text_results.extend(rec_texts)
                         elif isinstance(rec_texts, str):
                             full_text_results.append(rec_texts)
                         else:
                             text = line.get('text', line.get('rec_text', ''))
                             if text:
                                 full_text_results.append(text)
                         continue
                    
                    # Standard structure
                    if isinstance(line, (list, tuple)) and len(line) > 1:
                        if isinstance(line[1], (list, tuple)):
                             text = line[1][0]
                        else:
                             text = line[0] if isinstance(line[0], str) else str(line)
                        full_text_results.append(text)

            full_text_string = "\n".join(full_text_results)
            return (full_text_string,)
            
        except Exception as e:
            print(f"CRITICAL ERROR in PaddleOCR_Node: {e}")
            traceback.print_exc()
            raise RuntimeError(f"PaddleOCR Failed: {e}\n{traceback.format_exc()}")


class PaddleOCR_TestNode:
    """
    A simple test node that adds 1 to the input integer.
    Useful for verifying basic custom node functionality.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_input": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_output",)
    FUNCTION = "test_add"
    CATEGORY = "PaddleOCR"

    def test_add(self, int_input):
        return (int_input + 1,)


class PaddleOCR_Unified_Node:
    """
    Reviewer: User (Designer)
    Concept: Pure OCR Node (v5/v4/v3)
    A single node acting as a facade for standard PaddleOCR capabilities.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ocr_version": (["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"], {"default": "PP-OCRv5"}),
                "language": (["ch", "en", "japan", "korean", "chinese_cht", "french", "german"], {"default": "ch"}),
                "use_angle_cls": ("BOOLEAN", {"default": True, "label_on": "Enable Angle Classification", "label_off": "Disable"}),
            },
            "optional": {
                "use_tensorrt": ("BOOLEAN", {"default": False, "label_on": "Enable TensorRT (Faster)", "label_off": "Disable TensorRT"}),
                "precision": (["fp32", "fp16", "int8"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("text", "json_output")
    FUNCTION = "apply_unified_ocr"
    CATEGORY = "PaddleOCR"

    def apply_unified_ocr(self, image, ocr_version, language, use_angle_cls, use_tensorrt, precision):
        hw_kwargs = get_paddle_hw_kwargs()
        
        # Inject user overrides for high-end optimization
        if use_tensorrt:
            hw_kwargs["use_tensorrt"] = True
            hw_kwargs["precision"] = precision
            print(f"DEBUG: TensorRT Enabled with precision {precision}")
            
        print(f"DEBUG: Unified Node (Pure OCR) - Ver: {ocr_version}, Lang: {language}, Angle: {use_angle_cls}, HW: {hw_kwargs}")

        try:
            cv_images = tensor_to_cv2_img(image)
            results_txt = []
            results_json = []

            if PaddleOCR is None:
                    raise ImportError("PaddleOCR not installed.")
            
            # Standard Init
            ocr = PaddleOCR(ocr_version=ocr_version, lang=language, use_angle_cls=use_angle_cls, **hw_kwargs)
            
            for img_numpy in cv_images:
                result = ocr.ocr(img_numpy, cls=use_angle_cls) 
                # Result structure: [[[[x1,y1],[x2,y2]..], ("text", score)], ...]
                
                page_txt = []
                page_json = []
                
                if result:
                        # Handle batch wrapper if needed
                        if isinstance(result, list) and len(result)>0 and isinstance(result[0], list) and isinstance(result[0][0], list):
                            lines = result[0]
                        else:
                            lines = result
                        
                        for line in lines:
                            # line: [box, (text, score)]
                            if len(line) >= 2:
                                text_info = line[1]
                                text = text_info[0]
                                score = text_info[1]
                                box = line[0]
                                
                                page_txt.append(text)
                                
                                page_json.append({
                                    "text": text,
                                    "confidence": float(score),
                                    "box": box
                                })
                
                results_txt.append("\n".join(page_txt))
                results_json.append(page_json)

            # Final Aggregation
            final_txt = "\n\n".join(results_txt)
            
            # JSON needs to be serialize-safe
            import json
            try:
                final_json_str = json.dumps(results_json, ensure_ascii=False, indent=2)
            except:
                final_json_str = str(results_json)

            return (final_txt, final_json_str)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Unified OCR Failed: {e}")

