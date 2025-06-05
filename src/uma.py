"""
# Modal app for Fairchem's OMol25-trained UMA models

This app provides inference endpoints for the **Universal Materials Accelerator (UMA)** models
trained on the OMol25 dataset. These models support multiple domains including:

- **Catalysis** (Open Catalyst 2020)
- **Inorganic materials** 
- **Molecules**
- **MOFs** (Open Direct Air Capture)
- **Molecular crystals**

The UMA models use **Fairchem v2 API** with models loaded from a Modal volume.

## Key Features

- ✅ **Multi-domain support**: Single model handles all material types
- ✅ **Volume-based loading**: Models loaded from `omol25-model-weights` volume
- ✅ **Fairchem v2 compatible**: Uses modern `FAIRChemCalculator`
- ✅ **GPU acceleration**: Automatically detects and uses available GPUs
- ✅ **Comprehensive methods**: Energy/forces, relaxation, MD, spin gaps, etc.

## Available Models

- **UMA Small** (`uma_sm`): ~30M parameters, fastest inference

## Usage Example

```python
import modal
app = modal.App.lookup("fairchem-uma-omol25")
UMASmall = app.get_class("UMASmall")
uma_small = UMASmall()

# Predict energy and forces for a structure
structure = {
    "symbols": ["Fe", "Fe"],
    "positions": [[0.0, 0.0, 0.0], [1.43, 1.43, 1.43]],
    "cell": [[2.86, 0.0, 0.0], [0.0, 2.86, 0.0], [0.0, 0.0, 2.86]],
    "pbc": [True, True, True],
}

result = uma_small.predict.remote(structure, task_domain="omat")
print(f"Energy: {result['energy']:.4f} eV")
```
"""

import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union, Optional

import modal

# Create volume for storing UMA models
uma_models_volume = modal.Volume.from_name("omol25-model-weights")


class UMAModelSize(str, Enum):
    """Available UMA model sizes."""
    SMALL = "uma_sm"  # Small model
    # medium and large models are not yet available
    # check: https://huggingface.co/facebook/UMA#model-checkpoints
    # MEDIUM = "uma_md"  # Medium model  
    # LARGE = "uma_lg"  # Large model


class TaskDomain(str, Enum):
    """Supported task domains for UMA models."""
    CATALYSIS = "oc20"  # Catalysis (Open Catalyst 2020)
    MATERIALS = "omat"  # Inorganic materials
    MOLECULES = "omol"  # Molecules
    MOFS = "odac"  # MOFs (Open Direct Air Capture)
    MOLECULAR_CRYSTALS = "omc"  # Molecular crystals


@dataclass
class UMAModelInfo:
    """Information about a UMA model variant."""
    size: UMAModelSize
    description: str
    parameters: str
    recommended_domains: List[TaskDomain]
    volume_path: str  # Path within the volume


# Registry of available UMA models
UMA_MODELS = {
    UMAModelSize.SMALL: UMAModelInfo(
        size=UMAModelSize.SMALL,
        description="Small UMA model - fastest inference",
        parameters="~30M",
        recommended_domains=[
            TaskDomain.CATALYSIS,
            TaskDomain.MATERIALS,
            TaskDomain.MOLECULES,
            TaskDomain.MOFS,
            TaskDomain.MOLECULAR_CRYSTALS,
        ],
        volume_path="uma_sm.pt",
    ),
    # UMAModelSize.MEDIUM: UMAModelInfo(
    #     size=UMAModelSize.MEDIUM,
    #     description="Medium UMA model - balanced performance",
    #     parameters="~86M",
    #     recommended_domains=[
    #         TaskDomain.CATALYSIS,
    #         TaskDomain.MATERIALS,
    #         TaskDomain.MOLECULES,
    #         TaskDomain.MOFS,
    #         TaskDomain.MOLECULAR_CRYSTALS,
    #     ],
    #     volume_path="uma-medium",
    # ),
    # UMAModelSize.LARGE: UMAModelInfo(
    #     size=UMAModelSize.LARGE,
    #     description="Large UMA model - highest accuracy",
    #     parameters="~153M",
    #     recommended_domains=[
    #         TaskDomain.CATALYSIS,
    #         TaskDomain.MATERIALS,
    #         TaskDomain.MOLECULES,
    #         TaskDomain.MOFS,
    #         TaskDomain.MOLECULAR_CRYSTALS,
    #     ],
    #     volume_path="uma-large",
    # ),
}


# Create base image with Fairchem v2 and dependencies (no model downloads)
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "fairchem-core>=2.0.0",
        "ase>=3.22.1",
        "numpy",
    )
    .run_commands(
        # Install PyG dependencies for the specific torch version
        "pip install pyg_lib torch_geometric torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    )
    .env({
        "TORCH_HOME": "/root/.cache/torch",
    })
)

app = modal.App(name="fairchem-uma-omol25")


class BaseUMAModel:
    """
    # Base class for UMA model inference
    
    Provides common functionality for loading and running **Universal Materials Accelerator** models
    from Modal volumes using the Fairchem v2 API.
    
    ## Features
    
    - **Multi-domain prediction**: Supports oc20, omat, omol, odac, omc task domains
    - **Volume-based loading**: Loads models from Modal volumes without HF authentication
    - **GPU optimization**: Automatically uses GPU when available
    - **Comprehensive calculations**: Energy/forces, relaxation, MD, spin gaps, adsorption energies
    
    ## Supported Task Domains
    
    - `oc20`: **Catalysis** (Open Catalyst 2020)
    - `omat`: **Inorganic materials**
    - `omol`: **Molecules** 
    - `odac`: **MOFs** (Open Direct Air Capture)
    - `omc`: **Molecular crystals**
    """
    
    def __init__(self, model_size: UMAModelSize):
        self.model_size = model_size
        self.calculator = None
        self.model_info = UMA_MODELS[model_size]
    
    @modal.enter()
    def load_model(self):
        """
        # Load UMA model calculator from Modal volume
        
        Loads the UMA model checkpoint from the Modal volume using Fairchem's internal
        `load_predict_unit` function, which doesn't require Hugging Face authentication.
        
        ## Process
        
        1. **Locate checkpoint**: Finds model file in `/models/{volume_path}`
        2. **Load predictor**: Uses `load_predict_unit()` to create predictor from checkpoint
        3. **Create calculator**: Wraps predictor in `FAIRChemCalculator`
        4. **GPU detection**: Automatically uses GPU if available
        
        ## Error Handling
        
        - Raises `FileNotFoundError` if checkpoint not found in volume
        - Handles GPU/CPU device selection automatically
        """
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit
        
        print(f"Loading UMA model: {self.model_size.value}")
        
        # Load model directly from volume checkpoint
        model_path = pathlib.Path(f"/models/{self.model_info.volume_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint {self.model_size.value} not found in volume at {model_path}. "
                "Please ensure the model has been uploaded to the volume."
            )
        
        # Use load_predict_unit to load the model from checkpoint
        # This is the internal function used by get_predict_unit but doesn't require HF auth
        predictor = load_predict_unit(
            str(model_path),
            "default",
            None,
            "cuda" if self._has_gpu() else "cpu",
        )
        
        # Create the calculator with the predictor and default task
        self.calculator = FAIRChemCalculator(predictor, task_name="omat")
        
        print(f"Successfully loaded {self.model_size.value} from {model_path}")
    
    def _has_gpu(self):
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_calculator_for_task(self, task_domain: TaskDomain):
        """Get the calculator configured for the specified domain."""
        # UMA models are designed to handle multiple domains automatically
        # The task specificity is handled through the model's training
        # We can optionally update the task_name if the calculator supports it
        try:
            if hasattr(self.calculator, 'task_name'):
                self.calculator.task_name = task_domain.value
        except Exception:
            # If updating task_name fails, just use the calculator as-is
            pass
        
        return self.calculator
    
    def _validate_atoms(self, atoms_dict: Dict[str, Any]) -> Any:
        """Convert atoms dictionary to ASE Atoms object."""
        from ase import Atoms
        
        if isinstance(atoms_dict, dict):
            # Convert dictionary representation to ASE Atoms
            atoms = Atoms(
                symbols=atoms_dict.get("symbols", []),
                positions=atoms_dict.get("positions", []),
                cell=atoms_dict.get("cell"),
                pbc=atoms_dict.get("pbc", False),
            )
            
            # Add any additional properties
            if "charges" in atoms_dict:
                atoms.set_initial_charges(atoms_dict["charges"])
            if "magnetic_moments" in atoms_dict:
                atoms.set_initial_magnetic_moments(atoms_dict["magnetic_moments"])
            
            return atoms
        else:
            # Assume it's already an ASE Atoms object
            return atoms_dict
    
    def predict_energy_forces(
        self,
        structures: Union[Dict[str, Any], List[Dict[str, Any]]],
        task_domain: TaskDomain = TaskDomain.MATERIALS,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        # Predict energy and forces for structures
        
        Performs **energy and force predictions** for single structures or batches using the UMA model.
        
        ## Parameters
        
        - **structures**: Single structure dict or list of structure dicts
        - **task_domain**: Target domain (`oc20`, `omat`, `omol`, `odac`, `omc`)
        
        ## Structure Format
        
        ```python
        {
            "symbols": ["Fe", "Fe"],  # List of element symbols
            "positions": [[0,0,0], [1.43,1.43,1.43]],  # Atomic positions (Å)
            "cell": [[2.86,0,0], [0,2.86,0], [0,0,2.86]],  # Unit cell (Å)
            "pbc": [True, True, True]  # Periodic boundary conditions
        }
        ```
        
        ## Returns
        
        Dictionary (or list of dicts) containing:
        - **energy**: Potential energy in eV
        - **forces**: Atomic forces in eV/Å  
        - **success**: Whether prediction succeeded
        - **task_domain**: Domain used for prediction
        - **error**: Error message (if success=False)
        """
        calculator = self.get_calculator_for_task(task_domain)
        
        # Handle single structure
        if isinstance(structures, dict):
            try:
                atoms = self._validate_atoms(structures)
                atoms.calc = calculator
                
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                
                return {
                    "energy": float(energy),
                    "forces": forces.tolist(),
                    "success": True,
                    "task_domain": task_domain.value,
                }
            except Exception as e:
                return {
                    "energy": None,
                    "forces": None,
                    "success": False,
                    "error": str(e),
                    "task_domain": task_domain.value,
                }
        
        # Handle multiple structures
        results = []
        for structure in structures:
            try:
                atoms = self._validate_atoms(structure)
                atoms.calc = calculator
                
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                
                results.append({
                    "energy": float(energy),
                    "forces": forces.tolist(),
                    "success": True,
                    "task_domain": task_domain.value,
                })
            except Exception as e:
                results.append({
                    "energy": None,
                    "forces": None,
                    "success": False,
                    "error": str(e),
                    "task_domain": task_domain.value,
                })
        
        return results
    
    def relax(
        self,
        structure: Dict[str, Any],
        task_domain: TaskDomain = TaskDomain.MATERIALS,
        fmax: float = 0.05,
        steps: int = 200,
        optimizer: str = "FIRE",
    ) -> Dict[str, Any]:
        """
        # Relax structure to minimum energy configuration
        
        Performs **geometric optimization** to find the lowest energy structure using ASE optimizers.
        
        ## Parameters
        
        - **structure**: Structure dictionary (see `predict_energy_forces` for format)
        - **task_domain**: Target domain (`oc20`, `omat`, `omol`, `odac`, `omc`)
        - **fmax**: Force convergence criterion in eV/Å (default: 0.05)
        - **steps**: Maximum optimization steps (default: 200)
        - **optimizer**: Algorithm (`"FIRE"`, `"LBFGS"`, `"BFGS"`)
        
        ## Features
        
        - **Cell relaxation**: Automatically includes cell optimization for periodic systems
        - **Force convergence**: Stops when max force < fmax
        - **Progress tracking**: Reports initial/final energies and steps taken
        
        ## Returns
        
        Dictionary containing:
        - **initial_energy**: Starting energy in eV
        - **final_energy**: Optimized energy in eV
        - **energy_change**: Energy difference in eV
        - **initial_structure**: Original atomic structure
        - **final_structure**: Relaxed atomic structure  
        - **converged**: Whether optimization converged
        - **steps_taken**: Number of optimization steps
        - **success**: Whether relaxation succeeded
        """
        from ase.optimize import FIRE, LBFGS, BFGS
        from ase.filters import FrechetCellFilter
        import io
        import sys
        
        calculator = self.get_calculator_for_task(task_domain)
        
        try:
            atoms = self._validate_atoms(structure)
            atoms.calc = calculator
            
            # Store initial state
            initial_energy = atoms.get_potential_energy()
            initial_positions = atoms.get_positions().copy()
            initial_cell = atoms.get_cell().copy()
            
            # Choose optimizer
            if optimizer.upper() == "FIRE":
                opt_class = FIRE
            elif optimizer.upper() == "LBFGS":
                opt_class = LBFGS
            elif optimizer.upper() == "BFGS":
                opt_class = BFGS
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            # Set up optimization with cell relaxation for periodic systems
            if any(atoms.get_pbc()):
                # Use type: ignore to suppress the linter error - this is correct ASE usage
                dyn = opt_class(FrechetCellFilter(atoms))  # type: ignore
            else:
                dyn = opt_class(atoms)
            
            # Capture optimization output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                dyn.run(fmax=fmax, steps=steps)
                converged = dyn.get_number_of_steps() < steps
            finally:
                sys.stdout = old_stdout
            
            # Get final state
            final_energy = atoms.get_potential_energy()
            final_positions = atoms.get_positions()
            final_cell = atoms.get_cell()
            
            return {
                "initial_energy": float(initial_energy),
                "final_energy": float(final_energy),
                "energy_change": float(final_energy - initial_energy),
                "initial_structure": {
                    "symbols": atoms.get_chemical_symbols(),
                    "positions": initial_positions.tolist(),
                    "cell": initial_cell.tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                },
                "final_structure": {
                    "symbols": atoms.get_chemical_symbols(),
                    "positions": final_positions.tolist(),
                    "cell": final_cell.tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                },
                "converged": converged,
                "steps_taken": dyn.get_number_of_steps(),
                "success": True,
                "task_domain": task_domain.value,
                "optimizer": optimizer,
            }
            
        except Exception as e:
            return {
                "initial_energy": None,
                "final_energy": None,
                "energy_change": None,
                "initial_structure": None,
                "final_structure": None,
                "converged": False,
                "steps_taken": 0,
                "success": False,
                "error": str(e),
                "task_domain": task_domain.value,
                "optimizer": optimizer,
            }
    
    def run_molecular_dynamics(
        self,
        structure: Dict[str, Any],
        task_domain: TaskDomain = TaskDomain.MOLECULES,
        temperature_K: float = 300.0,
        timestep_fs: float = 0.1,
        steps: int = 1000,
        friction: float = 0.001,
        thermostat: str = "langevin",
    ) -> Dict[str, Any]:
        """
        Run molecular dynamics simulation.
        
        Args:
            structure: Structure as dictionary
            task_domain: Domain for the prediction
            temperature_K: Temperature in Kelvin
            timestep_fs: Timestep in femtoseconds
            steps: Number of MD steps
            friction: Friction coefficient for Langevin thermostat
            thermostat: Type of thermostat ("langevin", "nose-hoover")
            
        Returns:
            Dictionary containing:
            - trajectory: List of structures at each step
            - energies: List of energies at each step
            - temperatures: List of temperatures at each step
            - success: Whether simulation succeeded
        """
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.nvtberendsen import NVTBerendsen
        import numpy as np
        
        calculator = self.get_calculator_for_task(task_domain)
        
        try:
            atoms = self._validate_atoms(structure)
            atoms.calc = calculator
            
            # Set up MD
            if thermostat.lower() == "langevin":
                dyn = Langevin(
                    atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=temperature_K,
                    friction=friction / units.fs,
                )
            elif thermostat.lower() == "nose-hoover" or thermostat.lower() == "nvt":
                dyn = NVTBerendsen(
                    atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=temperature_K,
                    taut=100 * units.fs,  # Coupling time constant
                )
            else:
                raise ValueError(f"Unsupported thermostat: {thermostat}")
            
            # Storage for trajectory data
            trajectory = []
            energies = []
            temperatures = []
            
            # Run MD
            for step in range(steps):
                dyn.run(1)  # Run one step
                
                # Store data
                trajectory.append({
                    "symbols": atoms.get_chemical_symbols(),
                    "positions": atoms.get_positions().tolist(),
                    "cell": atoms.get_cell().tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                })
                
                energies.append(float(atoms.get_potential_energy()))
                temperatures.append(float(atoms.get_temperature()))
            
            return {
                "trajectory": trajectory,
                "energies": energies,
                "temperatures": temperatures,
                "final_structure": trajectory[-1],
                "average_energy": float(np.mean(energies)),
                "average_temperature": float(np.mean(temperatures)),
                "steps_completed": steps,
                "success": True,
                "task_domain": task_domain.value,
                "thermostat": thermostat,
            }
            
        except Exception as e:
            return {
                "trajectory": None,
                "energies": None,
                "temperatures": None,
                "final_structure": None,
                "average_energy": None,
                "average_temperature": None,
                "steps_completed": 0,
                "success": False,
                "error": str(e),
                "task_domain": task_domain.value,
                "thermostat": thermostat,
            }
    
    def calculate_spin_gap(
        self,
        structure: Dict[str, Any],
        spin_states: List[int],
        charges: Optional[List[int]] = None,
        task_domain: TaskDomain = TaskDomain.MOLECULES,
    ) -> Dict[str, Any]:
        """
        Calculate spin gap between different spin states.
        
        Args:
            structure: Base structure as dictionary
            spin_states: List of spin multiplicities to calculate
            charges: List of charges for each spin state (optional)
            task_domain: Domain for the prediction
            
        Returns:
            Dictionary containing:
            - energies: Dictionary of energies for each spin state
            - spin_gaps: Dictionary of energy differences
            - lowest_spin_state: Spin state with lowest energy
            - success: Whether calculation succeeded
        """
        calculator = self.get_calculator_for_task(task_domain)
        
        try:
            if charges is None:
                charges = [0] * len(spin_states)
            
            if len(charges) != len(spin_states):
                raise ValueError("Number of charges must match number of spin states")
            
            energies = {}
            
            # Calculate energy for each spin state
            for spin, charge in zip(spin_states, charges):
                atoms = self._validate_atoms(structure)
                
                # Set spin and charge information
                atoms.info.update({"spin": spin, "charge": charge})
                atoms.calc = calculator
                
                energy = atoms.get_potential_energy()
                energies[f"spin_{spin}"] = float(energy)
            
            # Calculate spin gaps
            spin_gaps = {}
            energy_values = list(energies.values())
            min_energy = min(energy_values)
            
            for i, (spin, energy) in enumerate(energies.items()):
                if energy != min_energy:
                    spin_gaps[f"{spin}_gap"] = float(energy - min_energy)
            
            # Find lowest energy spin state
            lowest_energy_spin = min(energies.keys(), key=lambda k: energies[k])
            
            return {
                "energies": energies,
                "spin_gaps": spin_gaps,
                "lowest_spin_state": lowest_energy_spin,
                "spin_states_calculated": spin_states,
                "charges_used": charges,
                "success": True,
                "task_domain": task_domain.value,
            }
            
        except Exception as e:
            return {
                "energies": None,
                "spin_gaps": None,
                "lowest_spin_state": None,
                "spin_states_calculated": spin_states,
                "charges_used": charges,
                "success": False,
                "error": str(e),
                "task_domain": task_domain.value,
            }
    
    def calculate_adsorption_energy(
        self,
        slab_structure: Dict[str, Any],
        adsorbate_structure: Dict[str, Any],
        combined_structure: Dict[str, Any],
        task_domain: TaskDomain = TaskDomain.CATALYSIS,
    ) -> Dict[str, Any]:
        """
        Calculate adsorption energy: E_ads = E_combined - E_slab - E_adsorbate.
        
        Args:
            slab_structure: Clean slab structure
            adsorbate_structure: Isolated adsorbate structure
            combined_structure: Slab + adsorbate structure
            task_domain: Domain for the prediction
            
        Returns:
            Dictionary containing:
            - adsorption_energy: Adsorption energy in eV
            - slab_energy: Energy of clean slab
            - adsorbate_energy: Energy of isolated adsorbate
            - combined_energy: Energy of combined system
            - success: Whether calculation succeeded
        """
        calculator = self.get_calculator_for_task(task_domain)
        
        try:
            # Calculate energy of clean slab
            slab_atoms = self._validate_atoms(slab_structure)
            slab_atoms.calc = calculator
            slab_energy = slab_atoms.get_potential_energy()
            
            # Calculate energy of isolated adsorbate
            adsorbate_atoms = self._validate_atoms(adsorbate_structure)
            adsorbate_atoms.calc = calculator
            adsorbate_energy = adsorbate_atoms.get_potential_energy()
            
            # Calculate energy of combined system
            combined_atoms = self._validate_atoms(combined_structure)
            combined_atoms.calc = calculator
            combined_energy = combined_atoms.get_potential_energy()
            
            # Calculate adsorption energy
            adsorption_energy = combined_energy - slab_energy - adsorbate_energy
            
            return {
                "adsorption_energy": float(adsorption_energy),
                "slab_energy": float(slab_energy),
                "adsorbate_energy": float(adsorbate_energy),
                "combined_energy": float(combined_energy),
                "success": True,
                "task_domain": task_domain.value,
            }
            
        except Exception as e:
            return {
                "adsorption_energy": None,
                "slab_energy": None,
                "adsorbate_energy": None,
                "combined_energy": None,
                "success": False,
                "error": str(e),
                "task_domain": task_domain.value,
            }
    
    def calculate_formation_energy(
        self,
        compound_structure: Dict[str, Any],
        reference_structures: Dict[str, Dict[str, Any]],
        stoichiometry: Dict[str, float],
        task_domain: TaskDomain = TaskDomain.MATERIALS,
    ) -> Dict[str, Any]:
        """
        Calculate formation energy from reference states.
        
        Args:
            compound_structure: Structure of the compound
            reference_structures: Dictionary of reference structures {element: structure}
            stoichiometry: Dictionary of stoichiometric coefficients {element: coefficient}
            task_domain: Domain for the prediction
            
        Returns:
            Dictionary containing:
            - formation_energy: Formation energy in eV
            - compound_energy: Energy of compound
            - reference_energies: Energies of reference states
            - success: Whether calculation succeeded
        """
        calculator = self.get_calculator_for_task(task_domain)
        
        try:
            # Calculate energy of compound
            compound_atoms = self._validate_atoms(compound_structure)
            compound_atoms.calc = calculator
            compound_energy = compound_atoms.get_potential_energy()
            
            # Calculate energies of reference states
            reference_energies = {}
            total_reference_energy = 0.0
            
            for element, structure in reference_structures.items():
                if element not in stoichiometry:
                    continue
                    
                ref_atoms = self._validate_atoms(structure)
                ref_atoms.calc = calculator
                ref_energy = ref_atoms.get_potential_energy()
                
                # Get energy per atom for this reference
                num_atoms = len(ref_atoms)
                energy_per_atom = ref_energy / num_atoms
                
                reference_energies[element] = {
                    "total_energy": float(ref_energy),
                    "energy_per_atom": float(energy_per_atom),
                    "num_atoms": num_atoms,
                }
                
                # Add contribution to total reference energy
                total_reference_energy += stoichiometry[element] * energy_per_atom
            
            # Calculate formation energy
            formation_energy = compound_energy - total_reference_energy
            
            return {
                "formation_energy": float(formation_energy),
                "compound_energy": float(compound_energy),
                "total_reference_energy": float(total_reference_energy),
                "reference_energies": reference_energies,
                "stoichiometry": stoichiometry,
                "success": True,
                "task_domain": task_domain.value,
            }
            
        except Exception as e:
            return {
                "formation_energy": None,
                "compound_energy": None,
                "total_reference_energy": None,
                "reference_energies": None,
                "stoichiometry": stoichiometry,
                "success": False,
                "error": str(e),
                "task_domain": task_domain.value,
            }


@app.cls(
    gpu="A10G", 
    image=base_image,
    volumes={"/models": uma_models_volume}
)
class UMASmall(BaseUMAModel):
    """
    # UMA Small model inference endpoint
    
    **Fast and efficient** UMA model with ~30M parameters, optimized for quick inference
    across all supported material domains.
    
    ## Key Features
    
    - 🚀 **Fastest inference**: ~30M parameter model optimized for speed
    - 🌍 **Multi-domain**: Supports catalysis, materials, molecules, MOFs, crystals
    - ⚡ **GPU accelerated**: Automatic GPU detection and usage
    - 🔄 **Comprehensive methods**: Energy prediction, relaxation, MD, spin gaps, etc.
    
    ## Usage
    
    ```python
    import modal
    app = modal.App.lookup("fairchem-uma-omol25")
    uma_small = UMASmall()
    
    # Basic energy/force prediction
    result = uma_small.predict.remote(structure, task_domain="omat")
    
    # Structure relaxation  
    relaxed = uma_small.relax.remote(structure, fmax=0.05)
    
    # Molecular dynamics
    md_result = uma_small.molecular_dynamics.remote(structure, steps=1000)
    ```
    """
    
    def __init__(self):
        super().__init__(UMAModelSize.SMALL)
    
    @modal.method()
    def predict(
        self,
        structures: Union[Dict[str, Any], List[Dict[str, Any]]],
        task_domain: str = "omat",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        # Predict energy and forces using UMA Small model
        
        **Primary endpoint** for energy and force predictions across all material domains.
        
        ## Parameters
        
        - **structures**: Single structure dict or list of structures
        - **task_domain**: Domain type - `"oc20"`, `"omat"`, `"omol"`, `"odac"`, or `"omc"`
        
        ## Returns
        
        Results with **energy** (eV), **forces** (eV/Å), and **success** status.
        
        ## Example
        
        ```python
        structure = {
            "symbols": ["Fe", "Fe"],
            "positions": [[0,0,0], [1.43,1.43,1.43]], 
            "cell": [[2.86,0,0], [0,2.86,0], [0,0,2.86]],
            "pbc": [True, True, True]
        }
        result = uma_small.predict.remote(structure, task_domain="omat")
        print(f"Energy: {result['energy']:.4f} eV")
        ```
        """
        domain = TaskDomain(task_domain)
        return self.predict_energy_forces(structures, domain)
    
    @modal.method()
    def relax_structure(
        self,
        structure: Dict[str, Any],
        task_domain: str = "omat",
        fmax: float = 0.05,
        steps: int = 200,
        optimizer: str = "FIRE",
    ) -> Dict[str, Any]:
        """
        Relax a structure using UMA Small model.
        
        Args:
            structure: Structure to relax
            task_domain: One of "oc20", "omat", "omol", "odac", "omc"
            fmax: Force convergence criterion
            steps: Maximum steps
            optimizer: Optimization algorithm
            
        Returns:
            Relaxation results
        """
        domain = TaskDomain(task_domain)
        return self.relax(structure, domain, fmax, steps, optimizer)
    
    @modal.method()
    def molecular_dynamics(
        self,
        structure: Dict[str, Any],
        task_domain: str = "omol",
        temperature_K: float = 300.0,
        timestep_fs: float = 0.1,
        steps: int = 1000,
        friction: float = 0.001,
        thermostat: str = "langevin",
    ) -> Dict[str, Any]:
        """
        Run molecular dynamics simulation using UMA Small model.
        
        Args:
            structure: Structure to simulate
            task_domain: One of "oc20", "omat", "omol", "odac", "omc"
            temperature_K: Temperature in Kelvin
            timestep_fs: Timestep in femtoseconds
            steps: Number of MD steps
            friction: Friction coefficient for Langevin thermostat
            thermostat: Type of thermostat ("langevin", "nvt")
            
        Returns:
            MD simulation results
        """
        domain = TaskDomain(task_domain)
        return self.run_molecular_dynamics(
            structure, domain, temperature_K, timestep_fs, steps, friction, thermostat
        )
    
    @modal.method()
    def spin_gap(
        self,
        structure: Dict[str, Any],
        spin_states: List[int],
        charges: Optional[List[int]] = None,
        task_domain: str = "omol",
    ) -> Dict[str, Any]:
        """
        Calculate spin gap between different spin states using UMA Small model.
        
        Args:
            structure: Base structure
            spin_states: List of spin multiplicities to calculate
            charges: List of charges for each spin state (optional)
            task_domain: One of "oc20", "omat", "omol", "odac", "omc"
            
        Returns:
            Spin gap calculation results
        """
        domain = TaskDomain(task_domain)
        return self.calculate_spin_gap(structure, spin_states, charges, domain)
    
    @modal.method()
    def adsorption_energy(
        self,
        slab_structure: Dict[str, Any],
        adsorbate_structure: Dict[str, Any],
        combined_structure: Dict[str, Any],
        task_domain: str = "oc20",
    ) -> Dict[str, Any]:
        """
        Calculate adsorption energy using UMA Small model.
        
        Args:
            slab_structure: Clean slab structure
            adsorbate_structure: Isolated adsorbate structure
            combined_structure: Slab + adsorbate structure
            task_domain: One of "oc20", "omat", "omol", "odac", "omc"
            
        Returns:
            Adsorption energy calculation results
        """
        domain = TaskDomain(task_domain)
        return self.calculate_adsorption_energy(
            slab_structure, adsorbate_structure, combined_structure, domain
        )
    
    @modal.method()
    def formation_energy(
        self,
        compound_structure: Dict[str, Any],
        reference_structures: Dict[str, Dict[str, Any]],
        stoichiometry: Dict[str, float],
        task_domain: str = "omat",
    ) -> Dict[str, Any]:
        """
        Calculate formation energy using UMA Small model.
        
        Args:
            compound_structure: Structure of the compound
            reference_structures: Dictionary of reference structures {element: structure}
            stoichiometry: Dictionary of stoichiometric coefficients {element: coefficient}
            task_domain: One of "oc20", "omat", "omol", "odac", "omc"
            
        Returns:
            Formation energy calculation results
        """
        domain = TaskDomain(task_domain)
        return self.calculate_formation_energy(
            compound_structure, reference_structures, stoichiometry, domain
        )

    @modal.method()
    def run_is2re_relaxation(
        self,
        atoms_dict: Dict[str, Any],
        task_domain: str = "omat",
        max_steps: int = 200, 
        force_max: float = 0.05,
        record_trajectory: bool = True,
        optimizer_type: str = "FIRE",
    ) -> Dict[str, Any]:
        """
        Run structure relaxation compatible with Matbench Discovery IS2RE task.

        Args:
            atoms_dict: Dictionary containing atoms data (ASE format).
            task_domain: Target domain ("oc20", "omat", "omol", "odac", "omc").
            max_steps: Maximum number of optimization steps.
            force_max: Force tolerance for convergence in eV/Å.
            record_trajectory: Whether to include trajectory frames in results.
            optimizer_type: Algorithm ("FIRE", "LBFGS", "BFGS").

        Returns:
            Dictionary containing:
            - final_energy: Relaxed energy.
            - relaxed_atoms: Relaxed structure (as dict).
            - trajectory: List of dictionaries with positions, cell, energy, forces, and stress for each step.
            - converged: Whether optimization converged.
            - nsteps: Number of optimization steps.
            - success: Whether the operation succeeded.
            - error: Error message if success is False.
        """
        from ase import Atoms
        from ase.optimize import FIRE, LBFGS, BFGS
        from ase.filters import FrechetCellFilter
        # numpy is imported at the top of the file and now in the image

        domain = TaskDomain(task_domain)
        calculator = self.get_calculator_for_task(domain)

        try:
            atoms = Atoms.fromdict(atoms_dict)
            atoms.calc = calculator

            if optimizer_type.upper() == "FIRE":
                opt_class = FIRE
            elif optimizer_type.upper() == "LBFGS":
                opt_class = LBFGS
            elif optimizer_type.upper() == "BFGS":
                opt_class = BFGS
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")

            if any(atoms.get_pbc()):
                optimizable_atoms = FrechetCellFilter(atoms)
            else:
                optimizable_atoms = atoms
            
            optimizer = opt_class(optimizable_atoms, logfile='-') # type: ignore[arg-type]

            trajectory_data = []
            if record_trajectory:
                def record_state():
                    cell_list = atoms.get_cell().tolist() # ASE Cell object has tolist()
                    trajectory_data.append({
                        'positions': atoms.get_positions().tolist(),
                        'cell': cell_list, 
                        'energy': float(atoms.get_potential_energy()),
                        'forces': atoms.get_forces().tolist(),
                        'stress': atoms.get_stress().tolist()
                    })
                optimizer.attach(record_state, interval=1)
                record_state()


            optimizer.run(fmax=force_max, steps=max_steps)
            
            if record_trajectory and optimizer.get_number_of_steps() > 0:
                current_pos = atoms.get_positions().tolist()
                if not trajectory_data or trajectory_data[-1]['positions'] != current_pos:
                    record_state()


            final_energy = float(atoms.get_potential_energy())
            relaxed_atoms_dict = atoms.todict()
            converged = optimizer.converged()
            nsteps = optimizer.get_number_of_steps()

            return {
                "final_energy": final_energy,
                "relaxed_atoms": relaxed_atoms_dict,
                "trajectory": trajectory_data if record_trajectory else None,
                "converged": converged,
                "nsteps": nsteps,
                "success": True,
            }

        except Exception as e:
            import traceback
            print(f"Error during IS2RE relaxation: {e}\\n{traceback.format_exc()}")
            return {
                "final_energy": None,
                "relaxed_atoms": None,
                "trajectory": None,
                "converged": False,
                "nsteps": 0,
                "success": False,
                "error": str(e),
            }


@app.local_entrypoint()
def main():
    """Test the UMA models with example structures."""
    
    # Example: Iron crystal (materials domain)
    iron_crystal = {
        "symbols": ["Fe", "Fe"],
        "positions": [[0.0, 0.0, 0.0], [1.43, 1.43, 1.43]],
        "cell": [[2.86, 0.0, 0.0], [0.0, 2.86, 0.0], [0.0, 0.0, 2.86]],
        "pbc": [True, True, True],
    }
    
    # Example: Water molecule (molecules domain)
    water_molecule = {
        "symbols": ["O", "H", "H"],
        "positions": [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [False, False, False],
    }
    
    print("Testing UMA models...")
    
    # Test UMA Small with materials
    print("\n=== UMA Small - Iron Crystal (Materials) ===")
    uma_small = UMASmall()
    result = uma_small.predict.remote(iron_crystal, task_domain="omat")
    if isinstance(result, dict) and result.get('success', False):
        print(f"Energy: {result['energy']:.4f} eV")
        print(f"Max force: {max(abs(f) for forces in result['forces'] for f in forces):.4f} eV/Å")
    else:
        error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'
        print(f"Prediction failed: {error_msg}")
    
    # Test UMA Small with molecules
    print("\n=== UMA Small - Water Molecule (Molecules) ===")
    result = uma_small.predict.remote(water_molecule, task_domain="omol")
    if isinstance(result, dict) and result.get('success', False):
        print(f"Energy: {result['energy']:.4f} eV")
        print(f"Max force: {max(abs(f) for forces in result['forces'] for f in forces):.4f} eV/Å")
    else:
        error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'
        print(f"Prediction failed: {error_msg}")
    
    # Test structure relaxation
    print("\n=== UMA Small - Structure Relaxation ===")
    relax_result = uma_small.relax_structure.remote(
        iron_crystal, 
        task_domain="omat",
        fmax=0.1,
        steps=50
    )
    if relax_result.get('success', False):
        print(f"Initial energy: {relax_result['initial_energy']:.4f} eV")
        print(f"Final energy: {relax_result['final_energy']:.4f} eV")
        print(f"Energy change: {relax_result['energy_change']:.4f} eV")
        print(f"Converged: {relax_result['converged']}")
        print(f"Steps taken: {relax_result['steps_taken']}")
    else:
        print(f"Relaxation failed: {relax_result.get('error', 'Unknown error')}")
    
    # Test molecular dynamics
    print("\n=== UMA Small - Molecular Dynamics (Water) ===")
    md_result = uma_small.molecular_dynamics.remote(
        water_molecule, 
        task_domain="omol",
        temperature_K=300.0,
        timestep_fs=0.1,
        steps=100,  # Short simulation for testing
        thermostat="langevin"
    )
    if md_result.get('success', False):
        print(f"MD completed: {md_result['steps_completed']} steps")
        print(f"Average energy: {md_result['average_energy']:.4f} eV")
        print(f"Average temperature: {md_result['average_temperature']:.1f} K")
    else:
        print(f"MD failed: {md_result.get('error', 'Unknown error')}")
    
    # Test spin gap calculation
    print("\n=== UMA Small - Spin Gap Calculation (CH2) ===")
    ch2_molecule = {
        "symbols": ["C", "H", "H"],
        "positions": [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-1.1, 0.0, 0.0]],
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [False, False, False],
    }
    
    spin_result = uma_small.spin_gap.remote(
        ch2_molecule,
        spin_states=[1, 3],  # singlet and triplet
        charges=[0, 0],
        task_domain="omol"
    )
    if spin_result.get('success', False):
        print(f"Energies: {spin_result['energies']}")
        print(f"Spin gaps: {spin_result['spin_gaps']}")
        print(f"Lowest energy state: {spin_result['lowest_spin_state']}")
    else:
        print(f"Spin gap calculation failed: {spin_result.get('error', 'Unknown error')}")
    
    # Test adsorption energy calculation
    print("\n=== UMA Small - Adsorption Energy (CO on Cu) ===")
    
    # Simple Cu slab - ensure consistent PBC for slab (periodic in x,y but not z)
    cu_slab = {
        "symbols": ["Cu", "Cu", "Cu", "Cu"],
        "positions": [[0.0, 0.0, 0.0], [2.56, 0.0, 0.0], [0.0, 2.56, 0.0], [2.56, 2.56, 0.0]],
        "cell": [[5.12, 0.0, 0.0], [0.0, 5.12, 0.0], [0.0, 0.0, 15.0]],
        "pbc": [True, True, True],  # Use fully periodic for compatibility
    }
    
    # CO molecule - make it fully non-periodic in a large cell
    co_molecule = {
        "symbols": ["C", "O"],
        "positions": [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]],
        "cell": [[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]],
        "pbc": [False, False, False],
    }
    
    # CO on Cu slab - match the slab PBC settings
    co_cu_slab = {
        "symbols": ["Cu", "Cu", "Cu", "Cu", "C", "O"],
        "positions": [
            [0.0, 0.0, 0.0], [2.56, 0.0, 0.0], [0.0, 2.56, 0.0], [2.56, 2.56, 0.0],
            [1.28, 1.28, 2.0], [1.28, 1.28, 3.13]
        ],
        "cell": [[5.12, 0.0, 0.0], [0.0, 5.12, 0.0], [0.0, 0.0, 15.0]],
        "pbc": [True, True, True],  # Match the slab PBC settings
    }
    
    ads_result = uma_small.adsorption_energy.remote(
        cu_slab, co_molecule, co_cu_slab, task_domain="oc20"
    )
    if ads_result.get('success', False):
        print(f"Adsorption energy: {ads_result['adsorption_energy']:.4f} eV")
        print(f"Slab energy: {ads_result['slab_energy']:.4f} eV")
        print(f"Adsorbate energy: {ads_result['adsorbate_energy']:.4f} eV")
    else:
        print(f"Adsorption energy calculation failed: {ads_result.get('error', 'Unknown error')}")
    
    print("\nUMA model testing completed!")
    print("\nNote: Make sure the UMA model checkpoint 'uma_sm.pt' is uploaded to the volume.")


if __name__ == "__main__":
    main()
