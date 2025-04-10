from main import run

from src.costs import B1HomogeneityCost
from src.costs.torch_cost_functions import TorchB1HomogeneityCostSAR
from src.data import Simulation
from src.utils import evaluate_coil_config

import numpy as np
import json

if __name__ == "__main__":
    # Load simulation data
    # simulation = Simulation("data/simulations/children_2_tubes_7_id_3012.h5")
    simulation = Simulation("data/simulations/children_0_tubes_2_id_19969.h5", "data/antenna/antenna.h5")
    
    # Define cost function
    cost_function = TorchB1HomogeneityCostSAR()
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function, timeout=300)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
