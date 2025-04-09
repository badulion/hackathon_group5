import time
import numpy as np
from tqdm import trange

from ..data.simulation import Simulation, CoilConfig
from ..costs.base import BaseCost

class HillClimbingOptimizer:
    """
    HillClimbingOptimizer uses a two-stage derivative-free optimization:
    
      1. Random Search: It randomly samples coil configurations and picks the best one.
      2. Local Refinement: It then improves the best configuration by applying small random perturbations.
      
    The goal is to maximize the cost function. The optimizer stops when either the maximum 
    iterations have been performed or the time limit (default 5 minutes) is reached.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 150,
                 refinement_iter: int = 500,
                 time_limit: float = 300) -> None:
        """
        Parameters:
            cost_function: An instance of a cost function (e.g., B1HomogeneityCost).
            max_iter: Maximum iterations for the random search phase.
            refinement_iter: Maximum iterations for the local refinement phase.
            time_limit: Maximum allowed time in seconds (default 300 seconds = 5 minutes).
        """
        self.cost_function = cost_function
        self.max_iter = max_iter
        self.refinement_iter = refinement_iter
        self.time_limit = time_limit
        self.direction = "maximize"  # We want to maximize the cost.

    def _sample_coil_config(self) -> CoilConfig:
        """
        Generate a random coil configuration:
          - Phase: 8 values uniformly in [0, 2π]
          - Amplitude: 8 values uniformly in [0, 1]
        """
        phase = np.random.uniform(0, 2 * np.pi, size=(8,))
        amplitude = np.random.uniform(0, 1, size=(8,))
        return CoilConfig(phase=phase, amplitude=amplitude)
    
    def _perturb(self, config: CoilConfig, std_phase: float, std_ampl: float) -> CoilConfig:
        """
        Generate a new configuration by adding normally distributed noise to the current configuration.
        The phase is wrapped into [0, 2π] and amplitude is clipped to [0, 1].
        """
        new_phase = config.phase + np.random.normal(0, std_phase, size=config.phase.shape)
        new_ampl = config.amplitude + np.random.normal(0, std_ampl, size=config.amplitude.shape)
        new_phase = np.mod(new_phase, 2 * np.pi)
        new_ampl = np.clip(new_ampl, 0, 1)
        return CoilConfig(phase=new_phase, amplitude=new_ampl)

    def optimize(self, simulation: Simulation) -> CoilConfig:
        """
        Runs the optimization:
          - In the Random Search phase, random configurations are generated and evaluated.
          - In the Local Refinement phase, the best candidate is perturbed to try to improve it.
        The procedure stops after the time limit is reached or after the maximum iterations.
        
        Returns:
            The best CoilConfig found.
        """
        start_time = time.time()
        # Initialize with a random configuration.
        best_config = self._sample_coil_config()
        best_cost = self.cost_function(simulation(best_config))
        
        pbar = trange(self.max_iter, desc="Random Search")
        for i in pbar:
            if time.time() - start_time >= self.time_limit:
                pbar.write("Time limit reached during random search.")
                break

            candidate = self._sample_coil_config()
            candidate_cost = self.cost_function(simulation(candidate))
            if candidate_cost > best_cost:
                best_cost = candidate_cost
                best_config = candidate
            pbar.set_postfix_str(f"Best Cost: {best_cost:.2f}")
        
        # Local refinement: gradually reduce the perturbation scale.
        std_phase = 0.1  # initial perturbation magnitude for phase
        std_ampl = 0.05  # initial perturbation magnitude for amplitude
        pbar2 = trange(self.refinement_iter, desc="Local Refinement")
        for _ in pbar2:
            if time.time() - start_time >= self.time_limit:
                pbar2.write("Time limit reached during refinement.")
                break

            candidate = self._perturb(best_config, std_phase, std_ampl)
            candidate_cost = self.cost_function(simulation(candidate))
            if candidate_cost > best_cost:
                best_cost = candidate_cost
                best_config = candidate
            # Optionally, reduce the perturbation size each iteration.
            std_phase *= 0.99
            std_ampl *= 0.99
            pbar2.set_postfix_str(f"Best Cost: {best_cost:.2f}")
        
        return best_config
