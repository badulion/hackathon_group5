from .data import CoilConfig, Simulation
from .costs.base import BaseCost

from typing import Dict, Any

def evaluate_coil_config(coil_config: CoilConfig, 
                         simulation: Simulation,
                         cost_function: BaseCost) -> Dict[str, Any]:
    """
    Evaluates the coil configuration using the cost function.

    Args:
        coil_config: Coil configuration to evaluate.
        simulation: Simulation object.
        cost_function: Cost function object.

    Returns:
        A dictionary containing the best coil configuration, cost, and cost improvement.
    """
    

    # Create a dictionary to store the results
    
    result = {
        "best_coil_phase": [float(i) for i in coil_config.phase],
        "best_coil_amplitude": [float(i) for i in coil_config.amplitude]
    }
    return result