"""
Fetal Brain Measurements Package

This package provides fetal brain measurement capabilities including:
- Brain segmentation
- Biometric measurements (CBD, BBD, TCD)
- Gestational age prediction
- Normative analysis
"""

__version__ = "1.0.0"
__author__ = "Fetal Brain Measurement Team"

# Make key classes available at package level
try:
    from .fetal_measure import FetalMeasure
    from .fetal_seg import FetalSegmentation
    __all__ = ['FetalMeasure', 'FetalSegmentation']
except ImportError:
    # Handle cases where dependencies aren't available
    __all__ = []
