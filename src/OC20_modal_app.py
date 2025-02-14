import pathlib
from dataclasses import dataclass
from enum import Enum
from sys import meta_path
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
    f"{ModelArchitecture.PAINN}_IS2RE": ModelInfo(
        name="PaiNN-IS2RE",
        registry_name="PaiNN-IS2RE-OC20-All",
        checkpoint_filename="painn_h1024_bs4x8_is2re_all.pt",
        description="PaiNN model for initial structure to relaxed energy",
        architecture=ModelArchitecture.PAINN,
        default_task=ModelTask.IS2RE,
    ),
    f"{ModelArchitecture.DIMENET_PLUS_PLUS}_IS2RE": ModelInfo(
        name="DimeNet++-IS2RE",
        registry_name="DimeNet++-IS2RE-OC20-All",
        checkpoint_filename="dimenetpp_all.pt",
        description="DimeNet++ model for initial structure to relaxed energy",
        architecture=ModelArchitecture.DIMENET_PLUS_PLUS,
        default_task=ModelTask.IS2RE,
    ),
    f"{ModelArchitecture.SCHNET}_IS2RE": ModelInfo(
        name="SchNet-IS2RE",
        registry_name="SchNet-IS2RE-OC20-All",
        checkpoint_filename="schnet_all.pt",
        description="SchNet model for initial structure to relaxed energy",
        architecture=ModelArchitecture.SCHNET,
        default_task=ModelTask.IS2RE,
    ),
}


class ModelCheckpointManager:
    """Manages downloading and accessing model checkpoints."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, architecture: ModelArchitecture) -> pathlib.Path:
        """Get the path for a specific model's checkpoint."""
        # Try direct lookup first (for S2EF models)
        model_info = MODELS.get(architecture)
        
        # If not found, try IS2RE variant
        if not model_info:
            is2re_key = f"{architecture}_IS2RE"
            model_info = MODELS.get(is2re_key)
        
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


class S2EFModel(BaseOC20Model):
    """Base class for Structure to Energy and Forces (S2EF) models.
    
    These models predict energy and forces for structures in their current configuration,
    without performing any optimization.
    """
    
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).

        Args:
            structures: Single structure (as a dictionary) or a list of structures.

        Returns:
            A dictionary (for a single structure) or a list of dictionaries (for multiple structures)
            containing:
                - energy: The potential energy of the structure (in eV)
                - forces: The forces on each atom (in eV/Å)
                - success: Boolean indicating if prediction succeeded
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
                try:
                    result = {
                        'energy': float(atoms.get_potential_energy()),
                        'forces': atoms.get_forces().tolist(),
                        'success': True
                    }
                except Exception as e:
                    result = {
                        'energy': None,
                        'forces': None,
                        'success': False,
                        'error': str(e)
                    }
                results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results[0] if single_input else results


class IS2REModel(BaseOC20Model):
    """Base class for Initial Structure to Relaxed Energy (IS2RE) models.
    
    These models directly predict the relaxed energy from an initial structure,
    without performing explicit optimization steps.
    """
    
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict relaxed energy for the given initial structure(s).

        Args:
            structures: Single structure (as a dictionary) or a list of structures.

        Returns:
            A dictionary (for a single structure) or a list of dictionaries (for multiple structures)
            containing:
                - relaxed_energy: The predicted relaxed energy (in eV)
                - success: Boolean indicating if prediction succeeded
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
                try:
                    # Note: The calculator should be trained for IS2RE prediction
                    # and return the relaxed energy directly
                    result = {
                        'relaxed_energy': float(atoms.get_potential_energy()),
                        'success': True
                    }
                except Exception as e:
                    result = {
                        'relaxed_energy': None,
                        'success': False,
                        'error': str(e)
                    }
                results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results[0] if single_input else results


class _EquiformerV2Large(S2EFModel):
    """Internal implementation of EquiformerV2 Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.EQUIFORMER_V2
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _GemNetOCLarge(S2EFModel):
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


class _PaiNNBase(S2EFModel):
    """Internal implementation of PaiNN model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.PAINN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _DimeNetPPLarge(S2EFModel):
    """Internal implementation of DimeNet++ Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.DIMENET_PLUS_PLUS
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _SchNetLarge(S2EFModel):
    """Internal implementation of SchNet Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCHNET
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _SCNLarge(S2EFModel):
    """Internal implementation of SCN Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.SCN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _ESCNLarge(S2EFModel):
    """Internal implementation of eSCN Large model."""
    
    def __init__(self):
        super().__init__()
        self.architecture = ModelArchitecture.ESCN
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


# IS2RE Model Implementations
class _PaiNN_IS2RE(IS2REModel):
    """Internal implementation of PaiNN model for IS2RE."""
    
    def __init__(self):
        super().__init__()
        self.architecture = f"{ModelArchitecture.PAINN}_IS2RE"
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
            only_output=['energy']  # Only request energy output for IS2RE models
        )


class _DimeNetPP_IS2RE(IS2REModel):
    """Internal implementation of DimeNet++ model for IS2RE."""
    
    def __init__(self):
        super().__init__()
        self.architecture = f"{ModelArchitecture.DIMENET_PLUS_PLUS}_IS2RE"
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


class _SchNet_IS2RE(IS2REModel):
    """Internal implementation of SchNet model for IS2RE."""
    
    def __init__(self):
        super().__init__()
        self.architecture = f"{ModelArchitecture.SCHNET}_IS2RE"
    
    def initialize_model(self, checkpoint_path: str):
        """Initialize the model from checkpoint."""
        import torch
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        
        self.checkpoint_path = checkpoint_path
        self.calculator = OCPCalculator(
            checkpoint_path=checkpoint_path,
            cpu=False if torch.cuda.is_available() else True,
        )


def download_specific_checkpoint(architecture: ModelArchitecture):
    """Download checkpoint for a specific model during image build."""
    import os
    
    CHECKPOINT_DIR = "/root/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.chmod(CHECKPOINT_DIR, 0o777)  # Ensure directory is writable
    
    manager = ModelCheckpointManager(CHECKPOINT_DIR)
    model_info = MODELS.get(architecture)
    if not model_info:
        raise ValueError(f"Unsupported model: {architecture.value}")
    
    try:
        from fairchem.core.models.model_registry import model_name_to_local_file
        print(f"Downloading checkpoint for {model_info.name}...")
        
        checkpoint_path = model_name_to_local_file(
            model_info.registry_name,
            local_cache=CHECKPOINT_DIR
        )
        
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint download failed for {model_info.name}")
            
        expected_path = manager.get_checkpoint_path(architecture)
        if checkpoint_path != expected_path:
            os.rename(checkpoint_path, expected_path)
            checkpoint_path = expected_path
            
        print(f"Successfully downloaded checkpoint to {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        print(f"Error downloading checkpoint for {model_info.name}: {str(e)}")
        raise

# Create base image with common dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "fairchem-core>=0.2.0",  # Ensure we have a recent version
        "ase>=3.22.1",
        "lmdb>=1.4.1",
        "requests>=2.31.0",  # For checkpoint downloads
        "tqdm>=4.66.1",      # For download progress
    )
    .run_commands(
        # Install PyG dependencies
        "pip install pyg_lib torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",
        # Create checkpoint directory with proper permissions
        "mkdir -p /root/checkpoints && chmod 777 /root/checkpoints"
    )
    .env({
        "TORCH_HOME": "/root/checkpoints",  # Set torch home for model downloads
        "FAIRCHEM_CACHE_DIR": "/root/checkpoints"  # Set fairchem cache location
    })
)

# Create specific images for each model architecture
equiformer_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.EQUIFORMER_V2,)
    )
)

gemnet_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.GEMNET_OC,)
    )
)

painn_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.PAINN,)
    )
)

dimenetpp_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.DIMENET_PLUS_PLUS,)
    )
)

schnet_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.SCHNET,)
    )
)

scn_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.SCN,)
    )
)

escn_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(ModelArchitecture.ESCN,)
    )
)

# Create IS2RE specific images
painn_is2re_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(f"{ModelArchitecture.PAINN}_IS2RE",)
    )
)

dimenetpp_is2re_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(f"{ModelArchitecture.DIMENET_PLUS_PLUS}_IS2RE",)
    )
)

schnet_is2re_image = (
    base_image
    .run_function(
        download_specific_checkpoint,
        args=(f"{ModelArchitecture.SCHNET}_IS2RE",)
    )
)

app = modal.App(name="opencatalyst-oc20")


class _BaseModal:
    """Base class for Modal interfaces to OC20 models.

    This class encapsulates the common behavior for loading model checkpoints and
    performing predictions. The core logic for prediction is implemented in the
    protected method `_predict`, which should be used by the concrete endpoint classes.
    """

    def __init__(self, model_cls: type[BaseOC20Model]):
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
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Perform prediction using the underlying model.

        This protected method implements the shared logic for predicting either:
         - Energy and forces (for S2EF models)
         - Relaxed energy (for IS2RE models)

        Args:
            structures: A single structure (as a dictionary) or a list of structures,
                      each represented as a dictionary of atomic attributes.

        Returns:
            For S2EF models:
                - energy: The potential energy (in eV)
                - forces: The atomic forces (in eV/Å)
                - success: Whether prediction succeeded
            
            For IS2RE models:
                - relaxed_energy: The predicted relaxed energy (in eV)
                - success: Whether prediction succeeded
        """
        return self.model.predict(structures)


def optimize_structure(
    atoms,
    calculator,
    steps: int = 200,
    fmax: float = 0.05,
) -> dict[str, Any]:
    """
    Optimize an atomic structure using ASE's BFGS optimizer.
    
    Args:
        atoms: ASE Atoms object to optimize
        calculator: Calculator that provides energy and forces
        steps: Maximum number of optimization steps
        fmax: Force convergence criterion in eV/Å
        
    Returns:
        Dictionary containing:
            - structure: Optimized structure
            - converged: Whether optimization converged
            - steps: Number of steps taken
            - energy: Final energy
            - forces: Final forces
            - success: Whether optimization succeeded
    """
    from ase.optimize import BFGS
    
    atoms.set_calculator(calculator)
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


@app.cls(gpu="A10G", image=equiformer_image)
class EquiformerV2_S2EF(_BaseModal):
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_EquiformerV2Large)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=gemnet_image)
class GemNetOC_S2EF(_BaseModal):
    """Modal interface to GemNet-OC model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_GemNetOCLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=painn_image)
class PaiNN_S2EF(_BaseModal):
    """Modal interface to PaiNN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_PaiNNBase)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=dimenetpp_image)
class DimeNetPP_S2EF(_BaseModal):
    """Modal interface to DimeNet++ model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_DimeNetPPLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=schnet_image)
class SchNet_S2EF(_BaseModal):
    """Modal interface to SchNet model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SchNetLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=scn_image)
class SCN_S2EF(_BaseModal):
    """Modal interface to SCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SCNLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


@app.cls(gpu="A10G", image=escn_image)
class ESCN_S2EF(_BaseModal):
    """Modal interface to eSCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_ESCNLarge)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict energy and forces for the given structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - energy: Structure energy in eV
                - forces: Atomic forces in eV/Å
                - success: Whether prediction succeeded
        """
        return self._predict(structures)
    
    @modal.method()
    def optimize(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Optimize structure(s) using this model as the calculator.
        
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
                - success: Whether optimization succeeded
        """
        single_input = isinstance(structures, dict)
        if single_input:
            structures = [structures]
            
        atoms_list = self.model._validate_structures(structures)
        results = []
        
        for atoms in atoms_list:
            result = optimize_structure(
                atoms,
                self.model.calculator,
                steps=steps,
                fmax=fmax
            )
            results.append(result)
            
        return results[0] if single_input else results


# IS2RE Modal Classes
@app.cls(gpu="A10G", image=painn_is2re_image)
class PaiNN_IS2RE(_BaseModal):
    """Modal interface to PaiNN model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_PaiNN_IS2RE)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict relaxed energy for the given initial structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - relaxed_energy: Predicted relaxed energy in eV
                - success: Whether prediction succeeded
        """
        return self._predict(structures)


@app.cls(gpu="A10G", image=dimenetpp_is2re_image)
class DimeNetPP_IS2RE(_BaseModal):
    """Modal interface to DimeNet++ model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_DimeNetPP_IS2RE)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict relaxed energy for the given initial structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - relaxed_energy: Predicted relaxed energy in eV
                - success: Whether prediction succeeded
        """
        return self._predict(structures)


@app.cls(gpu="A10G", image=schnet_is2re_image)
class SchNet_IS2RE(_BaseModal):
    """Modal interface to SchNet model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_SchNet_IS2RE)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict relaxed energy for the given initial structure(s).
        
        Args:
            structures: Single or list of ASE Atoms dictionary representations
        
        Returns:
            Single dictionary or list of dictionaries containing:
                - relaxed_energy: Predicted relaxed energy in eV
                - success: Whether prediction succeeded
        """
        return self._predict(structures)


@app.local_entrypoint()
def main():
    """Simple example demonstrating how to use OC20 models.
    
    Shows basic usage of:
    1. S2EF model for energy/forces prediction
    2. IS2RE model for direct relaxed energy prediction
    """
    # Create a simple test structure (H2O molecule)
    structure = create_simple_molecule()
    
    # Example 1: Using S2EF model
    print("\nExample 1: Structure to Energy and Forces (S2EF)")
    print("=" * 50)
    
    s2ef_model = PaiNN_S2EF()
    s2ef_result = s2ef_model.predict.remote(structure)
    
    if isinstance(s2ef_result, dict) and s2ef_result.get('success', False):
        print(f"Predicted energy: {s2ef_result['energy']:.3f} eV")
        forces = s2ef_result.get('forces', [])
        print(f"Predicted forces shape: {len(forces)} atoms × 3 components")
    else:
        error = s2ef_result.get('error', 'Unknown error') if isinstance(s2ef_result, dict) else str(s2ef_result)
        print(f"Prediction failed: {error}")
    
    # Example 2: Using IS2RE model
    print("\nExample 2: Initial Structure to Relaxed Energy (IS2RE)")
    print("=" * 50)
    
    is2re_model = PaiNN_IS2RE()
    is2re_result = is2re_model.predict.remote(structure)
    
    if isinstance(is2re_result, dict) and is2re_result.get('success', False):
        print(f"Predicted relaxed energy: {is2re_result['relaxed_energy']:.3f} eV")
    else:
        error = is2re_result.get('error', 'Unknown error') if isinstance(is2re_result, dict) else str(is2re_result)
        print(f"Prediction failed: {error}")

def create_simple_molecule():
    """Create a simple H2O molecule for testing."""
    from ase import Atoms
    
    # Create H2O molecule with typical bond lengths and angle
    water = Atoms('H2O',
                 positions=[[0.0, 0.0, 0.0],    # O
                          [0.0, 0.8, 0.6],      # H
                          [0.0, -0.8, 0.6]])    # H
    
    return water.todict()