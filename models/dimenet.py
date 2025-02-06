import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant


class _DimeNetPPLarge(OC20Model):
    """Internal implementation of DimeNet++ Large model.

    DimeNet++ is a directional message passing neural network that leverages
    directional information through spherical harmonics.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.DIMENET_PLUS_PLUS
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )
