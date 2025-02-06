from pathlib import Path
import modal
import numpy as np
from ase import Atoms
from typing import Dict, Union
import torch

CHECKPOINT_DIR = "/root/fairchem_checkpoints"
CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "Equiformer_V2_Large.pt"

def download_checkpoint():
    from fairchem.core.models.model_registry import model_name_to_local_file
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_name_to_local_file(
        "EquiformerV2-Large-S2EF-ODAC",
        local_cache=CHECKPOINT_DIR
    )
    assert Path(checkpoint_path).exists(), "Model checkpoint not found after download"
    print(f"Downloaded checkpoint to {checkpoint_path}")

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
    .run_function(download_checkpoint)
)

app.image = image

@app.cls(gpu="A10G")
class EquiformerV2_S2EF:
    """Interface to EquiformerV2 model for structure-to-energy-and-forces predictions."""

    @staticmethod
    def _atoms_to_dict(atoms: Atoms) -> Dict:
        """Convert ASE Atoms object to a dictionary representation."""
        return {
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.positions.tolist(),
            'cell': atoms.cell.tolist(),
            'pbc': atoms.pbc.tolist()
        }
    
    @staticmethod
    def _dict_to_atoms(data: Dict) -> Atoms:
        """Convert dictionary representation to ASE Atoms object."""
        return Atoms(
            symbols=data['symbols'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )

    @modal.enter()
    def load_model(self):
        """Load the EquiformerV2 model."""
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        self.calculator = OCPCalculator(
            checkpoint_path=CHECKPOINT_PATH,
            cpu=False if torch.cuda.is_available() else True,
        )

    @modal.method()
    def predict(
        self,
        structure: Union[Dict, Atoms],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> Dict:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: Either an ASE Atoms object or its dictionary representation
                      of the complete system (slab + adsorbate)
            
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        from ase.optimize import BFGS
        
        # Convert input to Atoms object if needed
        if isinstance(structure, dict):
            structure = self._dict_to_atoms(structure)
            
        # Validate input
        if len(structure) == 0:
            raise ValueError("Structure cannot be empty")
            
        # Setup calculator and run optimization
        structure.set_calculator(self.calculator)
        opt = BFGS(structure, trajectory=None)
        
        try:
            converged = opt.run(fmax=fmax, steps=steps)
            return {
                'structure': self._atoms_to_dict(structure),
                'converged': converged,
                'steps': opt.get_number_of_steps(),
                'energy': float(structure.get_potential_energy())
            }
        except Exception as e:
            return {
                'error': str(e),
                'structure': self._atoms_to_dict(structure)
            }

    @modal.method()
    def get_versions(self):
        """Get versions of key dependencies."""
        import torch
        return {
            'torch': torch.__version__,
            'cuda': torch.version.cuda if torch.cuda.is_available() else None
        }

@app.local_entrypoint()
def main():
    from ase.build import fcc111, add_adsorbate
    
    # Create a test case
    model = EquiformerV2_S2EF()
    print(model.get_versions.remote())
    
    # Create a Cu(111) surface
    slab = fcc111('Cu', size=(2, 2, 3), vacuum=10.0)
    
    # Create an H2 molecule and add it at a specific position
    adsorbate = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    
    # User controls exact placement - here we place it at a bridge site
    cell = slab.cell
    x = float(cell[0][0] / 3)  # 1/3 across the x direction
    y = float(cell[1][1] / 2)  # center in y
    add_adsorbate(slab, adsorbate, height=2.0, position=(x, y))
    
    # Convert to dictionary format
    structure_dict = EquiformerV2_S2EF._atoms_to_dict(slab)
    
    # Run prediction
    results = model.predict.remote(structure_dict)
    print(results)
