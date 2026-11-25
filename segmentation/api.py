"""Convenient in-process adapter when the suite root is on Python's import path.
Example: from segmentation.api import PlantSegSegmenter
Run ResShift in its separate process/environment to avoid dependency conflicts.
"""
from pathlib import Path
import sys
_local=str(Path(__file__).resolve().parent)
if _local not in sys.path:sys.path.insert(0,_local)
from plantseg_seg.predictor import PlantSegSegmenter,Prediction
__all__=['PlantSegSegmenter','Prediction']
