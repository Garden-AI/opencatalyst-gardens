from typing import Dict, Any, Optional, Union
from ase import Atoms
from ase.optimize import BFGS
from checkpoint.manager import ModelCheckpointManager, ModelArchitecture, ModelVariant

class OC20Model:
    """Base class for OC20 models."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """Initialize the base model.
        
        Args:
            checkpoint_dir: Directory for model checkpoints. If None, checkpoints
                          must be explicitly provided during model initialization.
        """
        self.checkpoint_manager = ModelCheckpointManager(checkpoint_dir) if checkpoint_dir else None
        self.architecture: Optional[ModelArchitecture] = None
        self.variant: Optional[ModelVariant] = None
        self.calculator = None

    def initialize_model(self, checkpoint_path: str):
        """Initialize the model with a checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement initialize_model")

    def predict(
        self,
        structure: Union[Dict[str, Any], Atoms],
        steps: int = 200,
        fmax: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Predict the optimized structure and energy.
        
        Args:
            structure: Either an ASE Atoms object or its dictionary representation
            steps: Maximum number of optimization steps
            fmax: Force convergence criterion in eV/Å
            
        Returns:
            Dictionary containing:
                - structure: Optimized atomic structure
                - converged: Whether optimization converged
                - steps: Number of optimization steps taken
                - energy: Final energy in eV
        """
        if self.calculator is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")
            
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

    @staticmethod
    def _atoms_to_dict(atoms: Atoms) -> Dict[str, Any]:
        """Convert ASE Atoms object to a dictionary representation."""
        return {
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.positions.tolist(),
            'cell': atoms.cell.array.tolist(),  # Use array property for proper numpy array
            'pbc': atoms.pbc.tolist()
        }
    
    @staticmethod
    def _dict_to_atoms(data: Dict[str, Any]) -> Atoms:
        """Convert dictionary representation to ASE Atoms object."""
        return Atoms(
            symbols=data['symbols'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        ) 