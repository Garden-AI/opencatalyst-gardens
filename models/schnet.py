import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant


class _SchNetLarge(OC20Model):
    """Internal implementation of SchNet Large model.

    SchNet is a continuous-filter convolutional neural network designed for
    modeling quantum interactions in molecules.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCHNET
        self.variant = ModelVariant.LARGE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )
