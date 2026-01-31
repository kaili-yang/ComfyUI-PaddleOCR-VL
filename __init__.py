# __init__.py
from .nodes import PaddleOCR_Node, PaddleOCR_TestNode

NODE_CLASS_MAPPINGS = {
    "PaddleOCR_TestNode": PaddleOCR_TestNode,
    "PaddleOCR_Node": PaddleOCR_Node, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddleOCR_TestNode": "PaddleOCR Test Node (Add 1)",
    "PaddleOCR_Node": "PaddleOCR Text Detection",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
