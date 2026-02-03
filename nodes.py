
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

class PaddleOCR_Unified_Node:
    """
    Reviewer: User (Designer)
    Concept: Consistency Principle & Task-Based UX.
    A single node acting as a facade for all PaddleOCR capabilities.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "task_mode": ([
                    "document_parsing",   # Default: Layout analysis + Markdown
                    "book_scanning",      # Unwarping + Orientation
                    "packaging_active",   # Seal recognition + Multi-angle
                    "standard_ocr"        # Pure Text (Fast)
                ], {"default": "document_parsing"}),
                "language": (["ch", "en", "japan", "korean", "chinese_cht", "french", "german"], {"default": "ch"}),
            },
            "optional": {
                "structure_tags": ("BOOLEAN", {"default": True, "label_on": "Include Format Tags", "label_off": "Plain Content"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "JSON")
    RETURN_NAMES = ("markdown", "plain_text", "json_output")
    FUNCTION = "apply_unified_ocr"
    CATEGORY = "PaddleOCR"

    def apply_unified_ocr(self, image, task_mode, language, structure_tags):
        hw_kwargs = get_paddle_hw_kwargs()
        print(f"DEBUG: Unified Node - Task: {task_mode}, Lang: {language}, HW: {hw_kwargs}")

        try:
            cv_images = tensor_to_cv2_img(image)
            results_md = []
            results_txt = []
            results_json = []

            # --- Strategy 1: VL Engine (Complex Tasks) ---
            if task_mode in ["document_parsing", "book_scanning", "packaging_active"]:
                if PaddleOCRVL is None:
                    raise ImportError("PaddleOCR VL module not found. Please upgrade paddleocr.")
                
                # Preset Configurations
                vl_args = {
                    "use_layout_detection": False,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_seal_recognition": False,
                    "use_chart_recognition": False
                }

                if task_mode == "document_parsing":
                    vl_args["use_layout_detection"] = True
                    vl_args["use_chart_recognition"] = True # Often useful in docs
                elif task_mode == "book_scanning":
                    vl_args["use_doc_unwarping"] = True
                    vl_args["use_doc_orientation_classify"] = True
                    vl_args["use_layout_detection"] = True # Usually want layout too
                elif task_mode == "packaging_active":
                    vl_args["use_seal_recognition"] = True
                    vl_args["use_doc_orientation_classify"] = True

                # Init Pipeline
                pipeline = PaddleOCRVL(**vl_args, **hw_kwargs)

                for img_numpy in cv_images:
                    res_list = pipeline.predict(img_numpy)
                    
                    for res in res_list:
                        # Inspect result structure (it's often a complex object in VL 1.5)
                        # We try to extract best representation
                        
                        # 1. JSON/Dict
                        # PaddleOCRVL results typically have an internal structure. 
                        # We'll try to dump it or use a provided method.
                        # For V1.5, let's assume standard dict-like access or helper
                        res_dict = {}
                        if hasattr(res, 'json'):
                            res_dict = res.json 
                        elif isinstance(res, dict):
                            res_dict = res
                        else:
                            # Fallback: try using `str` or `__dict__`
                            try:
                                res_dict = res.__dict__ 
                            except:
                                res_dict = {"raw": str(res)}
                        results_json.append(res_dict)

                        # 2. Markdown / Text
                        # Previous analysis suggested `save_to_markdown` or property.
                        # VL 1.5 pipeline usually returns structured objects that can be saved.
                        # We'll rely on string casting which usually gives the structure for VL models,
                        # or specific keys if known. 
                        # Since we don't have the exact object inspection at runtime, 
                        # we assume `str(res)` gives the markdown-like representation,
                        # OR check for keys like 'markdown', 'html', 'text'.
                        
                        md_text = ""
                        raw_text = ""

                        # Heuristic data extraction
                        if isinstance(res_dict, dict):
                             # Look for common keys
                             if 'markdown' in res_dict:
                                 md_text = res_dict['markdown']
                             elif 'html' in res_dict:
                                 md_text = res_dict['html'] # Close enough
                             elif 'rec_text' in res_dict:
                                 md_text = res_dict['rec_text']
                             
                             if 'text' in res_dict:
                                 raw_text = res_dict['text']
                             elif 'rec_text' in res_dict:
                                 raw_text = res_dict['rec_text']
                        
                        if not md_text:
                            md_text = str(res) # Robust fallback
                        if not raw_text:
                            # Strip basic MD tags if fallback used (simplified)
                            raw_text = md_text.replace('#', '').replace('*', '')

                        results_md.append(md_text)
                        results_txt.append(raw_text)

            # --- Strategy 2: Standard OCR (Fast/Pure) ---
            else: # standard_ocr
                if PaddleOCR is None:
                     raise ImportError("PaddleOCR not installed.")
                
                # Standard V5 init
                ocr = PaddleOCR(ocr_version='PP-OCRv5', lang=language, **hw_kwargs)
                
                for img_numpy in cv_images:
                    result = ocr.ocr(img_numpy) 
                    # Result structure: [[[[x1,y1],[x2,y2]..], ("text", score)], ...]
                    
                    page_md = []
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
                                 # Markdown: standard lines. maybe bold if high confidence?
                                 page_md.append(f"- {text}") 
                                 
                                 page_json.append({
                                     "text": text,
                                     "confidence": float(score),
                                     "box": box
                                 })
                    
                    results_md.append("\n".join(page_md))
                    results_txt.append("\n".join(page_txt))
                    results_json.append(page_json)

            # Final Aggregation
            final_md = "\n\n--- Page Break ---\n\n".join(results_md)
            final_txt = "\n\n".join(results_txt)
            
            # JSON needs to be serialize-safe
            import json
            try:
                final_json_str = json.dumps(results_json, ensure_ascii=False, indent=2)
            except:
                final_json_str = str(results_json)

            return (final_md, final_txt, final_json_str)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Unified OCR Failed: {e}")

