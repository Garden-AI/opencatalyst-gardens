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
    """Base class for OC20 models providing structure optimization functionality."""
    
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


class _EquiformerV2Large(BaseOC20Model):
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


class _GemNetOCLarge(BaseOC20Model):
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


class _PaiNNBase(BaseOC20Model):
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


class _DimeNetPPLarge(BaseOC20Model):
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


class _SchNetLarge(BaseOC20Model):
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


class _SCNLarge(BaseOC20Model):
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


class _ESCNLarge(BaseOC20Model):
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
class _PaiNN_IS2RE(BaseOC20Model):
    """Internal implementation of PaiNN model for IS2RE."""
    
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


class _DimeNetPP_IS2RE(BaseOC20Model):
    """Internal implementation of DimeNet++ model for IS2RE."""
    
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


class _SchNet_IS2RE(BaseOC20Model):
    """Internal implementation of SchNet model for IS2RE."""
    
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
        "lmdb",
    )
    .run_commands(
        "pip install pyg_lib torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )
    .run_function(download_checkpoints)
)

app = modal.App(name="opencatalyst-oc20", image=image)


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
        """Load and initialize the model with its checkpoint.

        This method retrieves the correct checkpoint for the configured model architecture,
        then initializes the model's calculator. It should be called before any predictions.
        """
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

        This protected method implements the shared logic for predicting optimized
        atomic structures, energies, and forces based on the provided input structures.
        It handles both single structure and batch predictions by:
         - Converting input structures (in dict or ASE Atoms format) to ASE Atoms objects.
         - Validating that no structure is empty.
         - Optimizing each structure using the model's associated calculator.
         - Returning the results in a consistent format.

        Args:
            structures: A single structure (as a dictionary) or a list of structures,
                        each represented as a dictionary of atomic attributes.
            steps: Maximum number of optimization steps to attempt.
            fmax: The force convergence criterion (in eV/Å). The optimization stops when
                  the maximum force on any atom is below this threshold.

        Returns:
            A dictionary (for a single structure) or a list of dictionaries (for multiple structures)
            where each dictionary includes:
                - 'structure': The optimized atomic configuration (as a dict).
                - 'converged': Boolean indicating if the optimization converged.
                - 'steps': Number of steps taken during optimization.
                - 'energy': The potential energy of the optimized structure (in eV).
                - 'forces': A list of force vectors (in eV/Å).
                - 'success': Boolean indicating overall success of the prediction.
                - An 'error' field, if any exceptions were encountered during optimization.
        """
        return self.model.predict(structures, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class EquiformerV2_S2EF(_BaseModal):
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions.

    This class serves as the public endpoint for clients calling the `predict` method.
    The method delegates to the base class's `_predict` implementation, ensuring that the
    service correctly associates the method with this specific class.
    """
    
    def __init__(self):
        super().__init__(_EquiformerV2Large)
    
    @modal.method()
    def predict(
        self,
        structures: dict[str, Any] | list[dict[str, Any]],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predict optimized atomic structures, energies, and forces.

        This method:
         - Accepts a single structure (or a list of structures) represented as a dictionary
           (or dictionaries) of atomic information.
         - Uses the underlying EquiformerV2 model to perform structure optimization via a
           shared prediction routine.
         - Returns the optimized structure, convergence status, number of optimization steps,
           energy, and forces.

        Args:
            structures: A dict or list of dicts representing ASE Atoms structures.
            steps: Maximum optimization steps to perform (default is 200).
            fmax: Force convergence threshold in eV/Å (default is 0.05).

        Returns:
            A dict or list of dicts containing the optimization results as described in the
            base class's `_predict` method.
        """
        return self._predict(structures, steps, fmax)


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
class PaiNN_S2EF(_BaseModal):
    """Modal interface to PaiNN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_PaiNNBase)
    
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
class DimeNetPP_S2EF(_BaseModal):
    """Modal interface to DimeNet++ model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_DimeNetPPLarge)
    
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
class SchNet_S2EF(_BaseModal):
    """Modal interface to SchNet model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SchNetLarge)
    
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
class SCN_S2EF(_BaseModal):
    """Modal interface to SCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_SCNLarge)
    
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
class ESCN_S2EF(_BaseModal):
    """Modal interface to eSCN model for structure-to-energy-and-forces predictions."""
    
    def __init__(self):
        super().__init__(_ESCNLarge)
    
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


# IS2RE Modal Classes
@app.cls(gpu="A10G")
class PaiNN_IS2RE(_BaseModal):
    """Modal interface to PaiNN model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_PaiNN_IS2RE)
    
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
class DimeNetPP_IS2RE(_BaseModal):
    """Modal interface to DimeNet++ model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_DimeNetPP_IS2RE)
    
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
class SchNet_IS2RE(_BaseModal):
    """Modal interface to SchNet model for initial-structure-to-relaxed-energy predictions."""
    
    def __init__(self):
        super().__init__(_SchNet_IS2RE)
    
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
    """Example usage demonstrating how to use OC20 models for catalysis predictions.

    Note: All models have the same interface, so you can use them interchangeably.
    Here we use the EquiformerV2 model.
    
    This example shows:
    1. Single structure prediction with structure optimization
    2. Batch prediction for parameter sweeps
    3. Model comparison
    """
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
    
    # 1. Single structure prediction
    slab = create_pt_co_slab()
    structure_dict = slab.todict()
    
    # Run structure optimization with EquiformerV2
    model = EquiformerV2_S2EF()
    result = model.predict.remote(
        structure_dict,
        steps=200,    # Maximum optimization steps
        fmax=0.05,    # Force convergence criterion in eV/Å
    )
    # Result contains:
    # - structure: Optimized structure as dictionary
    # - converged: Whether optimization converged
    # - steps: Number of steps taken
    # - energy: Final energy in eV
    print(result)
    
    # 2. Batch prediction (e.g., CO2 height sweep)
    batch_structures = []
    heights = [1.8, 2.0, 2.2, 2.4]  # Different CO2 heights to test
    for height in heights:
        slab = create_pt_co_slab()
        co2_indices = [-3, -2, -1]  # Last 3 atoms are CO2
        slab.positions[co2_indices] += [0, 0, height - 2.0]
        batch_structures.append(slab.todict())
    
    # Run batch prediction
    batch_results = model.predict.remote(batch_structures)
    # Each result contains:
    # - energy: Structure energy in eV
    # - forces: Atomic forces as list
    # - success: Whether prediction succeeded
    print(batch_results)
    
    # 3. Model comparison
    models = {
        "EquiformerV2": EquiformerV2_S2EF(),
        "GemNet-OC": GemNetOC_S2EF(),
        "PaiNN": PaiNN_S2EF(),
    }
    
    # Compare predictions for a single structure
    test_structure = batch_structures[1]  # Use 2.0Å height structure
    model_predictions = {
        name: model.predict.remote([test_structure])[0]
        for name, model in models.items()
    }
    # Each prediction contains energy and forces for comparison 
    print(model_predictions)