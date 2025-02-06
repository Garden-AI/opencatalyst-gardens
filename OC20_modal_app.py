import modal
import checkpoint
import models

app = modal.App(name="OpenCatalyst-OC20")

if modal.is_local():
    CHECKPOINT_DIR = "/root/fairchem_checkpoints"
else:
    CHECKPOINT_DIR = "/root/fairchem_checkpoints"


def download_checkpoints():
    """Download the default checkpoint during image build."""
    from checkpoint.manager import ModelCheckpointManager
    manager = ModelCheckpointManager(CHECKPOINT_DIR)
    # Download checkpoints for all supported models
    for arch, variant in manager.MODELS.keys():
        _ = manager.download_checkpoint(arch, variant)


app.image = (
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
    .add_local_python_source("models", "checkpoint")
)


class _Base:
    """Base class for Modal model interfaces."""

    def __init__(self):
        from checkpoint.manager import ModelCheckpointManager
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

    def _predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
        """Base implementation of predict method."""
        if self.model is None:
            raise RuntimeError("Model implementation not set")
        return self.model.predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class EquiformerV2_S2EF(_Base):
    """Modal interface to EquiformerV2 model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.equiformer import _EquiformerV2Large
        self.model = _EquiformerV2Large()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class GemNetOC_S2EF(_Base):
    """Modal interface to GemNet-OC model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.gemnet import _GemNetOCLarge
        self.model = _GemNetOCLarge()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class PaiNN_S2EF(_Base):
    """Modal interface to PaiNN model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.painn import _PaiNNBase
        self.model = _PaiNNBase()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class DimeNetPP_S2EF(_Base):
    """Modal interface to DimeNet++ model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.dimenet import _DimeNetPPLarge
        self.model = _DimeNetPPLarge()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class SchNet_S2EF(_Base):
    """Modal interface to SchNet model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.schnet import _SchNetLarge
        self.model = _SchNetLarge()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class SCN_S2EF(_Base):
    """Modal interface to SCN model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.scn import _SCNLarge
        self.model = _SCNLarge()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


@app.cls(gpu="A10G")
class ESCN_S2EF(_Base):
    """Modal interface to eSCN model for structure-to-energy-and-forces predictions."""

    def __init__(self):
        super().__init__()
        from models.scn import _ESCNLarge
        self.model = _ESCNLarge()

    @modal.method()
    def predict(
        self,
        structure,
        steps: int = 200,
        fmax: float = 0.05,
    ):
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
        return self._predict(structure, steps=steps, fmax=fmax)


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

    # Convert structure to dictionary for remote execution
    structure_dict = slab.todict()

    for name, model in models:
        print(f"\nTesting {name} model:")
        results = model.predict.remote(structure_dict)
        print(f"Results: {results}")

        # convert result back to Atoms if needed
        optimized_structure = Atoms.fromdict(results["structure"])
        print(optimized_structure)
