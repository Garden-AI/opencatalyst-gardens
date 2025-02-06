from pathlib import Path
import modal
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from typing import Dict, Union, Any

from checkpoint.manager import ModelCheckpointManager, ModelArchitecture, ModelVariant
from models.equiformer import _EquiformerV2Large

CHECKPOINT_DIR = "/root/fairchem_checkpoints"

def download_checkpoints():
    """Download the default checkpoint during image build."""
    manager = ModelCheckpointManager(CHECKPOINT_DIR)
    # Download checkpoints for all supported models
    for arch, variant in manager.MODELS.keys():
        manager.download_checkpoint(arch, variant)

app = modal.App(name="OpenCatalyst OC20")

image = (modal.Image.debian_slim(python_version="3.10")
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
    .run_function(download_checkpoints)
)

app.image = image

@app.cls(gpu="A10G")
class EquiformerV2Large_S2EF:
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions.
    
    This class provides the public Modal interface to the EquiformerV2 Large model,
    handling Modal-specific functionality while delegating core model operations to
    the internal implementation.
    """

    def __init__(self):
        self.model = _EquiformerV2Large()
        self.checkpoint_manager = ModelCheckpointManager(CHECKPOINT_DIR)

    @modal.enter()
    def load_model(self):
        """Load the EquiformerV2 model."""
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
            self.model.architecture,
            self.model.variant
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
            structure: Either an ASE Atoms object or its dictionary representation of the complete system (slab + adsorbate)
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
            
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)

    @staticmethod
    def atoms_to_dict(atoms: Atoms) -> Dict[str, Any]:
        """Convenience method for converting Atoms to dict format."""
        return _EquiformerV2Large._atoms_to_dict(atoms)


@app.local_entrypoint()
def main():
    """Example usage of the EquiformerV2 model."""
    # Create a test case
    model = EquiformerV2Large_S2EF()
    
    # Create a Cu(111) surface
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    
    # Create an H2 molecule
    adsorbate = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    
    # Add the adsorbate to the surface
    cell = slab.cell
    x = float(cell[0][0] / 3)  # 1/3 across the x direction
    y = float(cell[1][1] / 2)  # center in y
    add_adsorbate(slab, adsorbate, height=2.0, position=(x, y))
    
    # Convert to dictionary format and run prediction
    structure_dict = EquiformerV2Large_S2EF.atoms_to_dict(slab)
    results = model.predict.remote(structure_dict)
    print(results)
