import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import modal

class ModelArchitecture(str, Enum):
    """Available OC22 model architectures."""
    GEMNET_OC = "GemNet-OC"
    EQUIFORMER_V2 = "EquiformerV2"


class ModelTask(str, Enum):
    """Supported OC22 model tasks."""
    S2EF = "S2EF"  # Structure to energy and forces
    IS2RS = "IS2RS"  # Initial structure to relaxed structure


@dataclass
class ModelInfo:
    """Information about a specific model checkpoint."""
    name: str  # Display name
    registry_name: str  # Name to pass to model_name_to_local_file
    checkpoint_filename: str  # Expected filename after download
    description: str
    architecture: ModelArchitecture
    default_task: ModelTask


# Registry of available OC22 models with verified checkpoint information
# Note: Registry names and checkpoint filenames will need to be updated
MODELS = {
    ModelArchitecture.GEMNET_OC: ModelInfo(
        name="GemNet-OC",
        registry_name="GemNet-OC-S2EFS-OC20->OC22",  # Placeholder - needs update
        checkpoint_filename="gnoc_finetune_all_s2ef.pt",  # Placeholder - needs update
        description="GemNet-OC model trained on OC22 dataset for structure to energy and forces",
        architecture=ModelArchitecture.GEMNET_OC,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.EQUIFORMER_V2: ModelInfo(
        name="EquiformerV2",
        registry_name="EquiformerV2-lE4-lF100-S2EFS-OC22",
        checkpoint_filename="eq2_121M_e4_f100_oc22_s2ef.pt",
        description="EquiformerV2 model trained on OC222 dataset for structure to energt and force",
        architecture=ModelArchitecture.EQUIFORMER_V2,
        default_task=ModelTask.S2EF,
    ),
}



class ModelCheckpointManager:
    """Manages downloading and accessing model checkpoints."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, architecture: ModelArchitecture) -> pathlib.Path:
        """Get the path for a specific model's checkpoint."""
        model_info = MODELS.get(architecture)
        
        if not model_info:
            raise ValueError(f"Unsupported model: {architecture.value}")
        
        return self.checkpoint_dir / model_info.checkpoint_filename
    
    def download_checkpoint(self, architecture: ModelArchitecture) -> pathlib.Path:
        """Download a specific model checkpoint if needed."""
        from fairchem.core.models.model_registry import model_name_to_local_file
        
        model_info = MODELS.get(architecture)
        if not model_info:
            raise ValueError(f"Unsupported model: {architecture.value}")
        
        checkpoint_path = model_name_to_local_file(
            model_info.registry_name, local_cache=str(self.checkpoint_dir)
        )
        
        expected_path = self.get_checkpoint_path(architecture)
        
        # Handle case where downloaded file name doesn't match expected
        if pathlib.Path(checkpoint_path).exists() and checkpoint_path != expected_path:
            try:
                pathlib.Path(checkpoint_path).rename(expected_path)
                checkpoint_path = expected_path
                print(f"Renamed checkpoint to match expected filename: {expected_path}")
            except OSError as e:
                print(f"Warning: Could not rename checkpoint file: {e}")
        
        if not pathlib.Path(checkpoint_path).exists():
            raise RuntimeError(f"Failed to download checkpoint for {model_info.name}")
        
        print(f"Downloaded checkpoint to {checkpoint_path}")
        return pathlib.Path(checkpoint_path)


class BaseOC22Model:
    """Base class for OC22 models providing structure optimization functionality."""
    
    def __init__(self):
        self.architecture: ModelArchitecture
        self.calculator = None
        self.checkpoint_path = None

    def _validate_structures(
        self, 
        structures: list[dict[str, Any] | Any]
    ) -> list[Any]:
        """
        Convert structure representations to ASE Atoms objects and validate them.

        Args:
            structures: List of structures as dictionaries or ASE Atoms objects.

        Returns:
            List of ASE Atoms objects.

        Raises:
            ValueError: If any structure is empty.
        """
        from ase import Atoms

        validated = []
        for struct in structures:
            atoms_obj = Atoms.fromdict(struct) if isinstance(struct, dict) else struct
            if len(atoms_obj) == 0:
                raise ValueError("Structure cannot be empty")
            validated.append(atoms_obj)
        return validated

    def _optimize_structure(
        self, 
        atoms, 
        steps: int, 
        fmax: float
    ) -> dict[str, Any]:
        """
        Optimize the given ASE Atoms instance using the BFGS method.

        Args:
            atoms: ASE Atoms instance with its calculator attached.
            steps: Maximum optimization steps.
            fmax: Force convergence threshold (eV/Å).

        Returns:
            Dictionary containing the optimization result.
        """
        from ase.optimize import BFGS

        try:
            optimizer = BFGS(atoms, trajectory=None)
            converged = optimizer.run(fmax=fmax, steps=steps)
            result = {
                "structure": atoms.todict(),
                "converged": converged,
                "steps": optimizer.get_number_of_steps(),
                "energy": float(atoms.get_potential_energy()),
                "forces": atoms.get_forces().tolist(),
                "success": True,
            }
        except Exception as e:
            print(f"Optimization failed: {e}")
            result = {
                "structure": atoms.todict(),
                "converged": False,
                "steps": 0,
                "energy": None,
                "forces": None,
                "success": False,
                "error": str(e),
            }
        return result

    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize provided structure(s) and return the results.

        Args:
            structures: Single structure (dict) or list of structures.
            steps: Maximum optimization steps.
            fmax: Force convergence threshold (eV/Å).

        Returns:
            A dictionary for single input or a list of dictionaries for multiple structures.
        """
        import torch

        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]

        atoms_list = self._validate_structures(structures)
        results = []
        batch_size = 32
        
        for i in range(0, len(atoms_list), batch_size):
            batch = atoms_list[i : i + batch_size]
            for atoms in batch:
                atoms.set_calculator(self.calculator)
                results.append(self._optimize_structure(atoms, steps, fmax))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results[0] if single_input else results


# Model Implementation Classes
class _GemNetOCLarge(BaseOC22Model):
    """Internal implementation of GemNet-OC Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.GEMNET_OC
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _EquiformerV2(BaseOC22Model):
    """Internal implementation of EquiformerV2"""

    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.EQUIFORMER_V2

    def initialize_model(self, checkpoint_path: str):
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu= False if torch.cuda.is_available() else True,
        )


def download_checkpoints():
    """Download the default checkpoints during image build."""
    CHECKPOINT_DIR = "/root/checkpoints"
    manager = ModelCheckpointManager(CHECKPOINT_DIR)
    # Download checkpoints for all supported models
    for arch in MODELS.keys():
        _ = manager.download_checkpoint(arch)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "fairchem-core",
        "ase",
    )
    .run_commands(
        "pip install pyg_lib torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )
    .run_function(download_checkpoints)
)

app = modal.App(name="opencatalyst-oc22", image=image)


class _BaseModal:
    """Base class for Modal interfaces to OC22 models."""

    def __init__(self, model_cls: type[BaseOC22Model]):
        self.CHECKPOINT_DIR = "/root/checkpoints"
        self.checkpoint_manager = ModelCheckpointManager(self.CHECKPOINT_DIR)
        self.model = model_cls()

    @modal.enter()
    def load_model(self):
        """Load and initialize the model with its checkpoint."""
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
            self.model.architecture,
        )
        self.model.initialize_model(str(checkpoint_path))

    def _predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int,
        fmax: float,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Perform the structure optimization prediction using the underlying model.

        Args:
            structures: Single structure (dict) or list of structures.
            steps: Maximum optimization steps.
            fmax: Force convergence threshold (eV/Å).

        Returns:
            A dictionary for single input or a list of dictionaries for multiple structures.
        """
        return self.model.predict(structures, steps=steps, fmax=fmax)


# Modal Endpoints
@app.cls(gpu="A10G")
class GemNetOC_S2EF(_BaseModal):
    """Modal interface to GemNet-OC model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_GemNetOCLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict optimized structure(s) and energy/forces.
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - structure: Optimized atomic structure
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
                - forces: Final forces in eV/Å
        """
        return self._predict(structures, steps, fmax)


@app.cls(gpu="A10G")
class EquiformerV2_S2EF(_BaseModal):
    def __init__(self):
        super().__init__(_EquiformerV2)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict optimized structure(s) and energy/forces.
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - structure: Optimized atomic structure
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
                - forces: Final forces in eV/Å
        """
        return self._predict(structures, steps, fmax)


@app.local_entrypoint()
def main():
    """Example usage demonstrating how to use OC22 models for catalysis predictions."""
    from ase.build import fcc111, add_adsorbate
    from ase import Atoms
    
    def create_pt_co_slab():
        """Create a Pt-Co(111) slab with CO2 adsorbate."""
        # Create a Pt-Co(111) slab
        slab = fcc111('Pt', size=(2, 2, 4), vacuum=12.0)
        slab.symbols[2] = 'Co'  # Replace one second layer atom with Co
        
        # Create CO2 adsorbate
        co2 = Atoms('CO2',
                   positions=[[0.0, 0.0, 0.0],  # C
                            [0.0, 0.0, 1.16],   # O
                            [0.0, 0.0, -1.16]]) # O
        
        # Add CO2 at a bridge site
        add_adsorbate(slab, co2, height=2.0, position=(2.0, 1.5))
        return slab
    
    # Example using GemNet-OC model
    slab = create_pt_co_slab()
    structure_dict = slab.todict()
    
    # Run structure optimization
    model = GemNetOC_S2EF()
    result = model.predict.remote(
        structure_dict,
        steps=200,    # Maximum optimization steps
        fmax=0.05,    # Force convergence criterion in eV/Å
    )
    print(result)
    
    # Compare predictions from different models
    models = {
        "GemNet-OC": GemNetOC_S2EF(),
        "EquiformerV2": EquiformerV2_S2EF(),
    }
    
    model_predictions = {
        name: model.predict.remote([structure_dict])[0]
        for name, model in models.items()
    }
    print(model_predictions) 