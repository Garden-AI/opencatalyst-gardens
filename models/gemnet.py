import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant


class _GemNetOCLarge(OC20Model):
    """Internal implementation of GemNet-OC Large model.

    GemNet-OC is a geometric message passing neural network optimized for catalysis,
    with explicit geometric information in message passing.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.GEMNET_OC
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )
