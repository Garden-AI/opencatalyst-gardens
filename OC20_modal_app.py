from pathlib import Path
import modal
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from typing import Dict, Union, Any

from checkpoint.manager import ModelCheckpointManager, ModelArchitecture, ModelVariant
from models.base import OC20Model
from models.equiformer import _EquiformerV2Large
from models.gemnet import _GemNetOCLarge
from models.painn import _PaiNNBase

CHECKPOINT_DIR = "/root/fairchem_checkpoints"


def download_checkpoints():
    """Download the default checkpoint during image build."""
    manager = ModelCheckpointManager(CHECKPOINT_DIR)
    # Download checkpoints for all supported models
    for arch, variant in manager.MODELS.keys():
        manager.download_checkpoint(arch, variant)


app = modal.App(name="OpenCatalyst OC20")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch >= 2.4.0",
    )
    .run_commands(
        "pip install pyg_lib torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )
    .pip_install(
        "fairchem-core",
        "ase",
    )
    .run_function(download_checkpoints, force_build=True)
    .add_local_python_source("models", "checkpoint", "utils")
)

app.image = image


class _Base:
    """Base class for Modal model interfaces."""
    
    def __init__(self):
        self.checkpoint_manager = ModelCheckpointManager(CHECKPOINT_DIR)
        self.model = None  # Set by subclasses

    @modal.enter()
    def load_model(self):
        """Load the model."""
        if self.model is None:
            raise RuntimeError("Model implementation not set")
            
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
            self.model.architecture, self.model.variant
        )
        self.model.initialize_model(str(checkpoint_path))

    @modal.method()
    def predict(
        self,
        structure: Union[Dict[str, Any], Atoms],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Predict the optimized structure and energy.

        Args:
            structure: Either an ASE Atoms object or its dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å

        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary (from Atoms.todict())
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        if self.model is None:
            raise RuntimeError("Model implementation not set")
            
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class EquiformerV2_S2EF(_Base):
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        self.model = _EquiformerV2Large()


@app.cls(gpu="A10G")
class GemNetOC_S2EF(_Base):
    """Modal interface to GemNet-OC model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        self.model = _GemNetOCLarge()


@app.cls(gpu="A10G")
class PaiNN_S2EF(_Base):
    """Modal interface to PaiNN model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        self.model = _PaiNNBase()


@app.local_entrypoint()
def main():
    """Example usage demonstrating all available models."""

    # Create a catalyst slab
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
    adsorbate = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    # Add the adsorbate to the surface
    cell = slab.cell
    x = float(cell.array[0][0] / 3)  # 1/3 across the x direction
    y = float(cell.array[1][1] / 2)  # center in y
    add_adsorbate(slab, adsorbate, height=2.0, position=(x, y))

    # Test each model
    models = [
        ("EquiformerV2", EquiformerV2_S2EF()),
        ("GemNet-OC", GemNetOC_S2EF()),
        ("PaiNN", PaiNN_S2EF()),
    ]

    # Convert structure to dictionary for remote execution
    structure_dict = slab.todict()

    for name, model in models:
        print(f"\nTesting {name} model:")
        results = model.predict.remote(structure_dict)
        print(f"Results: {results}")

        # convert result back to Atoms if needed
        optimized_structure = Atoms.fromdict(results["structure"])
