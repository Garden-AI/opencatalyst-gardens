import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant

class _EquiformerV2Large(OC20Model):
    """Internal implementation of EquiformerV2 Large model for structure-to-energy-and-forces predictions.
    
    This is the internal implementation class that handles the core model functionality.
    The public interface is provided by the Modal-decorated class in modal_app.py.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.EQUIFORMER_V2
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        ) 