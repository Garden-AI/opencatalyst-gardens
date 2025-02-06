import torch
from models.base import OC20Model
from checkpoint.manager import ModelArchitecture, ModelVariant


class _PaiNNBase(OC20Model):
    """Internal implementation of PaiNN (Polarizable Interaction Neural Network) model.

    PaiNN is designed to learn molecular potentials while respecting physical symmetries,
    with particular focus on polarization effects.
    """

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.PAINN
        self.variant = ModelVariant.BASE

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )
