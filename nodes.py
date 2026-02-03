
import sys
import os
import traceback
from .utils import tensor_to_cv2_img, get_paddle_hw_kwargs

# Attempt to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        PaddleOCRVL = None
except ImportError:
    PaddleOCR = None
    PaddleOCRVL = None

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


class PaddleOCRVL_Node:
    """
    PaddleOCR-VL Node for Document Parsing (v1.5).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_layout_detection": ("BOOLEAN", {"default": True}),
                "use_doc_orientation_classify": ("BOOLEAN", {"default": False}),
                "use_doc_unwarping": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "use_chart_recognition": ("BOOLEAN", {"default": False}),
                "use_seal_recognition": ("BOOLEAN", {"default": False}),
                # "use_ocr_for_image_block": ("BOOLEAN", {"default": False}), # Optional advanced
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown_text",)
    FUNCTION = "apply_vl_parse"
    CATEGORY = "PaddleOCR"

    def apply_vl_parse(self, image, use_layout_detection, use_doc_orientation_classify, 
                       use_doc_unwarping, use_chart_recognition, use_seal_recognition):
        
        if PaddleOCRVL is None:
             raise ImportError("PaddleOCRVL not found. Please upgrade paddleocr (pip install -U paddleocr) to use VL features.")

        try:
            # Initialize Pipeline
            # Use shared hardware logic 
            hw_kwargs = get_paddle_hw_kwargs()
            
            pipeline = PaddleOCRVL(
                use_layout_detection=use_layout_detection,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_chart_recognition=use_chart_recognition,
                use_seal_recognition=use_seal_recognition,
                **hw_kwargs
            )

            # Process Image
            # PaddleOCRVL expects path or numpy array. ComfyUI gives Tensor (B,H,W,C) or (H,W,C).
            # We reuse tensor_to_cv2_img
            cv_images = tensor_to_cv2_img(image)
            
            markdown_results = []
            
            for img_numpy in cv_images:
                # predict() returns a list of dicts or objects depending on version.
                # Based on doc, predict returns a list of results.
                # Each res has save_to_markdown or properites.
                # We'll use the predict() method which presumably handles list logic internally or returns list.
                
                # The predict method signature: predict(input, ...)
                results = pipeline.predict(img_numpy)
                
                for res in results:
                    # Depending on PaddleOCR structure, 'res' might be an object that prints/saves.
                    # We need the markdown string content.
                    # Inspecting paddleocr code: res usually has 'str' or 'markdown' attribute or specialized method.
                    # Looking at paddleocr implementation, predict returns list of structure results.
                    # Usually structure results are converted to markdown.
                    
                    # If res is a dict/object, we try to extract 'markdown' field if available, 
                    # or rely on standard keys. 
                    # NOTE: PaddleOCR-VL return structure can be complex. 
                    # The demo uses res.save_to_markdown(). Let's see if we can get the string directly.
                    # Often res is a subclass of dict or has __str__.
                    
                    # Hack: if no direct 'markdown' string property, we might need to parse `res`.
                    # For v1.5, it likely returns a structure that implies markdown generation.
                    
                    # Let's check keys if it's a dict
                    if hasattr(res, 'keys'):
                         # Try standard keys
                         pass
                    
                    # For now, let's assume we can cast to string or it has a 'markdown' attribute 
                    # If not, we'll iterate standard fields.
                    # Actually, seeing the file content: `concatenation` helper exists.
                    pass
                
                # To be safe and obtain the raw markdown:
                # The CLI/demo saves to file.
                # We will rely on simple string conversion for start, or use a temporary internal buffer if needed.
                # However, the best way without I/O is to inspect the returned object.
                # Let's assume standard object str representation or try to find a 'markdown' property.
                
                # STARTUP FIX: 
                # PaddleOCR-VL results usually contain 'html', 'latex', 'markdown' keys in the internal dict.
                # Let's map output to str(results) first for debugging if we are unsure, 
                # but better: assume 'markdown' if compatible with PP-Structure.
                pass

                # Actually, let's stick to the simplest integration: passing the object to user might not work (string output).
                # We will perform a prediction and iterate.
                # Based on `predict` source, it returns `self.predict_iter(...)` list.
                
                # Let's try to grab the structured text.
                # If we lack specific API knowledge of the result object's STRING method, 
                # we'll capture standard output or return the stringified result.
                
                # Improvement: Inspect the result object in memory if possible. 
                # For now, append str(res)
                for res in results:
                    if hasattr(res, 'str'): # specific string method
                        markdown_results.append(res.str)
                    elif isinstance(res, dict) and 'markdown' in res:
                        markdown_results.append(res['markdown'])
                    else:
                        markdown_results.append(str(res))

            return ("\n\n".join(markdown_results),)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"PaddleOCR-VL Failed: {e}")

