from .base import OC20Model
from .equiformer import _EquiformerV2Large
from .gemnet import _GemNetOCLarge
from .painn import _PaiNNBase

# Note: The public Modal interface classes are defined in modal_app.py
__all__ = ["OC20Model"]  # Only export the base class
