from ..data.simulation import Simulation, SimulationData, CoilConfig
from skimage.measure import label, regionprops

# from ..costs.base import BaseCost
# from typing import Callable
from einops import rearrange, repeat, einsum
import numpy as np
import torch
import tqdm

from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhaseShift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
        self.amplitude = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))

    def forward(self, field):
        re_phase = torch.cos(self.phase) * self.amplitude
        im_phase = torch.sin(self.phase) * self.amplitude
        coeffs_real = torch.stack((re_phase, -im_phase), dim=0)
        coeffs_im = torch.stack((im_phase, re_phase), dim=0)
        coeffs = torch.stack((coeffs_real, coeffs_im), dim=0)
        coeffs = repeat(coeffs, "reimout reim coils -> hf reimout reim coils", hf=2)
        field_shift = einsum(
            field,
            coeffs,
            "hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...",
        )
        return field_shift


class TorchOptimizer:

    def __init__(self, cost_function, max_iter: int = 1000) -> None:
        self.cost_fuction = cost_function
        self.max_iter = max_iter

    def field_and_prop(self, sim_data: Simulation):
        labeled_mask = label(sim_data.simulation_raw_data.subject)
        regions = regionprops(labeled_mask)
        coords = regions[0].bbox
        field_np = sim_data.simulation_raw_data.field[
            :,
            :,
            :,
            int(coords[0]) : int(coords[3]),
            int(coords[1]) : int(coords[4]),
            int(coords[2]) : int(coords[5]),
            :,
        ]
        prop_np = sim_data.simulation_raw_data.properties[
            :,
            int(coords[0]) : int(coords[3]),
            int(coords[1]) : int(coords[4]),
            int(coords[2]) : int(coords[5]),
        ]

        field = torch.tensor(field_np, dtype=torch.float32).to(device)
        prop = torch.tensor(prop_np, dtype=torch.float32).to(device)

        return field, prop

    def optimize(self, simulation: Simulation):
        field, prop = self.field_and_prop(simulation)
        model = PhaseShift().to(device)
        optmizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        pbar = trange(self.max_iter)
        for i in pbar:
            optmizer.zero_grad()
            shifted_field = model(field)
            loss = self.cost_fuction.calc_cost(shifted_field, prop)
            loss.backward()
            optmizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")

        coil_config = CoilConfig(
            phase=model.phase.cpu().detach().numpy(),
            amplitude=model.amplitude.cpu().detach().numpy(),
        )
        return coil_config
