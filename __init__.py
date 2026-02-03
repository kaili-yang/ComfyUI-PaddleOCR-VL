# __init__.py
from .nodes import PaddleOCR_Node, PaddleOCR_TestNode, PaddleOCRVL_Node, PaddleOCR_Unified_Node

NODE_CLASS_MAPPINGS = {
    "PaddleOCR_Node": PaddleOCR_Node,
    "PaddleOCR_TestNode": PaddleOCR_TestNode,
    "PaddleOCRVL_Node": PaddleOCRVL_Node,
    "PaddleOCR_Unified_Node": PaddleOCR_Unified_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCR_Node": "PaddleOCR Text Detection",
    "PaddleOCR_TestNode": "PaddleOCR Test (Add+1)",
    "PaddleOCRVL_Node": "PaddleOCR VL (Doc Parser)",
    "PaddleOCR_Unified_Node": "PaddleOCR Unified (All-in-One)"
}


WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
