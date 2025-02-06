from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from enum import Enum, auto


class ModelArchitecture(Enum):
    """Available model architectures."""

    EQUIFORMER_V2 = "EquiformerV2"
    GEMNET_OC = "GemNet-OC"
    ESCN = "eSCN"
    PAINN = "PaiNN"
    SCHNET = "SchNet"
    DIMENET_PLUS_PLUS = "DimeNet++"


class ModelTask(Enum):
    """Supported model tasks."""

    S2EF = "S2EF"  # Structure to energy and forces
    IS2RE = "IS2RE"  # Initial structure to relaxed energy


class ModelVariant(Enum):
    """Model size/variant."""

    LARGE = "Large"
    BASE = "Base"
    SMALL = "Small"


@dataclass
class ModelInfo:
    """Information about a specific model checkpoint."""

    name: str  # Display name
    registry_name: str  # Name to pass to model_name_to_local_file
    checkpoint_filename: str  # Expected filename after download
    description: str
    architecture: ModelArchitecture
    variant: ModelVariant
    default_task: ModelTask


class ModelCheckpointManager:
    """Manages downloading and accessing model checkpoints."""

    # Registry of available models with verified checkpoint information
    MODELS = {
        # EquiformerV2 Models
        (ModelArchitecture.EQUIFORMER_V2, ModelVariant.LARGE): ModelInfo(
            name="EquiformerV2-Large",
            registry_name="EquiformerV2-153M-S2EF-OC20-All+MD",
            checkpoint_filename="Equiformer_V2_Large.pt",
            description="EquiformerV2 Large model for structure to energy and forces",
            architecture=ModelArchitecture.EQUIFORMER_V2,
            variant=ModelVariant.LARGE,
            default_task=ModelTask.S2EF,
        ),
        # GemNet-OC Models
        (ModelArchitecture.GEMNET_OC, ModelVariant.LARGE): ModelInfo(
            name="GemNet-OC-Large",
            registry_name="GemNet-OC-Large-S2EF-OC20-All+MD",
            checkpoint_filename="gemnet_oc_large_s2ef_all_md.pt",
            description="GemNet-OC model for structure to energy and forces",
            architecture=ModelArchitecture.GEMNET_OC,
            variant=ModelVariant.LARGE,
            default_task=ModelTask.S2EF,
        ),
        # PaiNN Models
        (ModelArchitecture.PAINN, ModelVariant.BASE): ModelInfo(
            name="PaiNN",
            registry_name="PaiNN-S2EF-OC20-All",
            checkpoint_filename="painn_h512_s2ef_all.pt",
            description="PaiNN model for structure to energy and forces",
            architecture=ModelArchitecture.PAINN,
            variant=ModelVariant.BASE,
            default_task=ModelTask.S2EF,
        ),
    }

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self, architecture: ModelArchitecture, variant: ModelVariant = None
    ) -> Path:
        """Get the path for a specific model's checkpoint."""
        if variant is None:
            variant = self._get_default_variant(architecture)

        model_key = (architecture, variant)
        model_info = self.MODELS.get(model_key)
        if not model_info:
            raise ValueError(
                f"Unsupported model: {architecture.value} {variant.value if variant else ''}"
            )
        return self.checkpoint_dir / model_info.checkpoint_filename

    def download_checkpoint(
        self, architecture: ModelArchitecture, variant: ModelVariant = None
    ) -> Path:
        """Download a specific model checkpoint if needed."""
        from fairchem.core.models.model_registry import model_name_to_local_file

        if variant is None:
            variant = self._get_default_variant(architecture)

        model_key = (architecture, variant)
        model_info = self.MODELS.get(model_key)
        if not model_info:
            raise ValueError(
                f"Unsupported model: {architecture.value} {variant.value if variant else ''}"
            )

        checkpoint_path = model_name_to_local_file(
            model_info.registry_name, local_cache=str(self.checkpoint_dir)
        )

        expected_path = self.get_checkpoint_path(architecture, variant)

        # Handle case where downloaded file name doesn't match expected
        if Path(checkpoint_path).exists() and checkpoint_path != expected_path:
            try:
                Path(checkpoint_path).rename(expected_path)
                checkpoint_path = expected_path
                print(f"Renamed checkpoint to match expected filename: {expected_path}")
            except OSError as e:
                print(f"Warning: Could not rename checkpoint file: {e}")

        if not Path(checkpoint_path).exists():
            raise RuntimeError(f"Failed to download checkpoint for {model_info.name}")

        print(f"Downloaded checkpoint to {checkpoint_path}")
        return Path(checkpoint_path)

    @classmethod
    def list_available_models(cls) -> Dict[tuple, str]:
        """List all available models and their descriptions."""
        return {
            (info.architecture, info.variant): info.description
            for info in cls.MODELS.values()
        }

    @classmethod
    def _get_default_variant(cls, architecture: ModelArchitecture) -> ModelVariant:
        """Get the default variant for a given architecture."""
        variants = [key[1] for key in cls.MODELS.keys() if key[0] == architecture]
        if not variants:
            raise ValueError(f"No variants available for {architecture.value}")
        # Prefer LARGE > BASE > SMALL
        if ModelVariant.LARGE in variants:
            return ModelVariant.LARGE
        if ModelVariant.BASE in variants:
            return ModelVariant.BASE
        return variants[0]
