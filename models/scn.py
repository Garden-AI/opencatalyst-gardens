import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant


class _SCNLarge(OC20Model):
    """Internal implementation of SCN Large model.

    Spherical Channel Network (SCN) is designed to learn representations of
    molecular systems using spherical harmonic channels.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCN
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _ESCNLarge(OC20Model):
    """Internal implementation of eSCN Large model.

    Enhanced Spherical Channel Network (eSCN) is an improved version of SCN
    with better performance and efficiency.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.ESCN
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )
