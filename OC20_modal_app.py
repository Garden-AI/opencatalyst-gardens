import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import modal


class ModelArchitecture(str, Enum):
    """Available model architectures."""
    EQUIFORMER_V2 = "EquiformerV2"
    GEMNET_OC = "GemNet-OC"
    ESCN = "eSCN"
    PAINN = "PaiNN"
    SCHNET = "SchNet"
    DIMENET_PLUS_PLUS = "DimeNet++"
    SCN = "SCN"


class ModelTask(str, Enum):
    """Supported model tasks."""
    S2EF = "S2EF"  # Structure to energy and forces
    IS2RE = "IS2RE"  # Initial structure to relaxed energy


@dataclass
class ModelInfo:
    """Information about a specific model checkpoint."""
    name: str  # Display name
    registry_name: str  # Name to pass to model_name_to_local_file
    checkpoint_filename: str  # Expected filename after download
    description: str
    architecture: ModelArchitecture
    default_task: ModelTask


# Registry of available models with verified checkpoint information
MODELS = {
    ModelArchitecture.EQUIFORMER_V2: ModelInfo(
        name="EquiformerV2-Large",
        registry_name="EquiformerV2-153M-S2EF-OC20-All+MD",
        checkpoint_filename="equiformer_v2_large.pt",
        description="EquiformerV2 Large model for structure to energy and forces",
        architecture=ModelArchitecture.EQUIFORMER_V2,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.GEMNET_OC: ModelInfo(
        name="GemNet-OC-Large",
        registry_name="GemNet-OC-Large-S2EF-OC20-All+MD",
        checkpoint_filename="gemnet_oc_large.pt",
        description="GemNet-OC model for structure to energy and forces",
        architecture=ModelArchitecture.GEMNET_OC,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.PAINN: ModelInfo(
        name="PaiNN",
        registry_name="PaiNN-S2EF-OC20-All",
        checkpoint_filename="painn_base.pt",
        description="PaiNN model for structure to energy and forces",
        architecture=ModelArchitecture.PAINN,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.DIMENET_PLUS_PLUS: ModelInfo(
        name="DimeNet++",
        registry_name="DimeNet++-S2EF-OC20-All",
        checkpoint_filename="dimenetpp_large.pt",
        description="DimeNet++ model for structure to energy and forces",
        architecture=ModelArchitecture.DIMENET_PLUS_PLUS,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.SCHNET: ModelInfo(
        name="SchNet",
        registry_name="SchNet-S2EF-OC20-All",
        checkpoint_filename="schnet_large.pt",
        description="SchNet model for structure to energy and forces",
        architecture=ModelArchitecture.SCHNET,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.SCN: ModelInfo(
        name="SCN",
        registry_name="SCN-S2EF-OC20-All+MD",
        checkpoint_filename="scn_large.pt",
        description="SCN model for structure to energy and forces",
        architecture=ModelArchitecture.SCN,
        default_task=ModelTask.S2EF,
    ),
    ModelArchitecture.ESCN: ModelInfo(
        name="eSCN",
        registry_name="eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",
        checkpoint_filename="escn_large.pt",
        description="eSCN model for structure to energy and forces",
        architecture=ModelArchitecture.ESCN,
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


class BaseOC20Model:
    """Base class for OC20 models."""
    
    def __init__(self):
        self.architecture: ModelArchitecture
        self.calculator = None
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_model")
    
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """Run structure optimization and return results."""
        from ase import Atoms
        from ase.optimize import BFGS
        
        if self.calculator is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")
        
        if isinstance(structure, dict):
            structure = Atoms.fromdict(structure)
        
        # Validate input
        if len(structure) == 0:
            raise ValueError("Structure cannot be empty")
        
        # Setup calculator and run optimization
        structure.set_calculator(self.calculator)
        optimizer = BFGS(structure, trajectory=None)
        
        try:
            converged = optimizer.run(fmax=fmax, steps=steps)
            return {
                "structure": structure.todict(),
                "converged": converged,
                "steps": optimizer.get_number_of_steps(),
                "energy": float(structure.get_potential_energy()),
            }
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {"error": str(e), "structure": structure.todict()}


class _EquiformerV2Large(BaseOC20Model):
    """Internal implementation of EquiformerV2 Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.EQUIFORMER_V2
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _GemNetOCLarge(BaseOC20Model):
    """Internal implementation of GemNet-OC Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.GEMNET_OC
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _PaiNNBase(BaseOC20Model):
    """Internal implementation of PaiNN model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.PAINN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _DimeNetPPLarge(BaseOC20Model):
    """Internal implementation of DimeNet++ Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.DIMENET_PLUS_PLUS
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _SchNetLarge(BaseOC20Model):
    """Internal implementation of SchNet Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCHNET
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _SCNLarge(BaseOC20Model):
    """Internal implementation of SCN Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _ESCNLarge(BaseOC20Model):
    """Internal implementation of eSCN Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.ESCN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
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

app = modal.App(name="opencatalyst-oc20", image=image)


class _BaseModal:
    """Base class for Modal interfaces to OC20 models."""
    
    def __init__(self, model_cls: BaseOC20Model):
        self.CHECKPOINT_DIR = "/root/checkpoints"
        self.checkpoint_manager = ModelCheckpointManager(self.CHECKPOINT_DIR)
        self.model = model_cls()
    
    @modal.enter()
    def load_model(self):
        """Load the model."""
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path(
            self.model.architecture,
        )
        self.model.initialize_model(str(checkpoint_path))


@app.cls(gpu="A10G")
class EquiformerV2_S2EF(_BaseModal):
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_EquiformerV2Large)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class GemNetOC_S2EF(_BaseModal):
    """Modal interface to GemNet-OC model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_GemNetOCLarge)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class PaiNN_S2EF(_BaseModal):
    """Modal interface to PaiNN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_PaiNNBase)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class DimeNetPP_S2EF(_BaseModal):
    """Modal interface to DimeNet++ model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_DimeNetPPLarge)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class SchNet_S2EF(_BaseModal):
    """Modal interface to SchNet model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SchNetLarge)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class SCN_S2EF(_BaseModal):
    """Modal interface to SCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SCNLarge)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class ESCN_S2EF(_BaseModal):
    """Modal interface to eSCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_ESCNLarge)
    
    @modal.method()
    def predict(
        self,
        structure: dict[str, Any],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: ASE Atoms object dictionary representation (from Atoms.todict())
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
        
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure as dictionary
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.local_entrypoint()
def main():
    """Example usage demonstrating all available models."""
    from ase.build import fcc111, add_adsorbate
    from ase import Atoms
    
    # Create a catalyst slab
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
    adsorbate = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    
    # Add the adsorbate to the surface
    cell = slab.cell
    x = float(cell.array[0][0] / 3)  # 1/3 across the x direction
    y = float(cell.array[1][1] / 2)  # center in y
    add_adsorbate(slab, adsorbate, height=2.0, position=(x, y))
    
    # Convert structure to dictionary for remote execution
    structure_dict = slab.todict()
    
    # Test each model
    models = [
        ("EquiformerV2", EquiformerV2_S2EF()),
        ("GemNet-OC", GemNetOC_S2EF()),
        ("PaiNN", PaiNN_S2EF()),
        ("DimeNet++", DimeNetPP_S2EF()),
        ("SchNet", SchNet_S2EF()),
        ("SCN", SCN_S2EF()),
        ("eSCN", ESCN_S2EF()),
    ]
    
    for name, model in models:
        print(f"\nTesting {name} model:")
        results = model.predict.remote(structure_dict)
        print(f"Results: {results}")
        
        # Convert result back to Atoms if needed
        optimized_structure = Atoms.fromdict(results["structure"])
        print(optimized_structure) 