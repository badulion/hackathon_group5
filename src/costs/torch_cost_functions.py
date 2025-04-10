from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator, SARCalculator

import numpy as np
import torch


class TorchB1HomogeneityCost:
    def __init__(self) -> None:
        super().__init__()

    def b1_calc(self, field):
        b_field = field[1]
        b_field_complex = b_field[0] + 1j * b_field[1]
        return 0.5 * (b_field_complex[0] + 1j * b_field_complex[1])

    def b1_homogeneity_cost(self, field):
        b1_plus = self.b1_calc(field)
        b1_plus_magnitude = torch.abs(b1_plus)
        b1_plus_mean = torch.mean(b1_plus_magnitude)
        b1_plus_std = torch.std(b1_plus_magnitude)
        return b1_plus_mean / (b1_plus_std + 1e-6)

    def calc_loss(self, field):
        return -1 * self.b1_homogeneity_loss(field)


class TorchB1HomogeneityCostSAR:
    def __init__(self) -> None:
        super().__init__()

    def sars_calc(self, field, properties):
        e_field = field[0]
        abs_efield_sq = torch.sum(e_field**2, axis=(0, 1))

        # get the conductivity and density tensors
        conductivity = properties[0]
        density = properties[2]

        return conductivity * abs_efield_sq / density

    def sar_cost(self, field, properties):
        return torch.max(self.sars_calc(field, properties))

    def calc_loss(self, field, prop, lambda_param=0.01):
        b1_homogeneity_cost = TorchB1HomogeneityCost()
        return b1_homogeneity_cost.b1_homogeneity_cost(
            field
        ) + lambda_param * self.sar_cost(field, prop)
