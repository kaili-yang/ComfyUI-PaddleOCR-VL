# __init__.py
from .nodes import PaddleOCR_Node, PaddleOCR_Unified_Node

NODE_CLASS_MAPPINGS = {
    "PaddleOCR_Node": PaddleOCR_Node,
    "PaddleOCR_Unified_Node": PaddleOCR_Unified_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCR_Node": "PaddleOCR (Lagacy)",
    "PaddleOCR_Unified_Node": "PaddleOCR Unified (All-in-One)"
}




__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
