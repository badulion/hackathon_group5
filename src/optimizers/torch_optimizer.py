from ..data.simulation import Simulation, SimulationData, CoilConfig
from skimage.measure import label, regionprops

from tqdm import trange
import torch
import lightning.pytorch as pl
from einops import repeat, einsum
from skimage.measure import label, regionprops
from torch.utils.data import DataLoader, Dataset

from src.data.simulation import Simulation
from src.costs.torch_cost_functions import TorchB1HomogeneityCostSAR


class Module(pl.LightningModule):
    def __init__(self):
        super(Module, self).__init__()
        self.phase = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
        self.amplitude = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
        self.loss_fn = TorchB1HomogeneityCostSAR()

    def _shift_phase(self, field):
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
    
    def _field_and_prop(self, field, properties, mask):
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        coords = regions[0].bbox
        field_np = field[
            :,
            :,
            :,
            int(coords[0]) : int(coords[3]),
            int(coords[1]) : int(coords[4]),
            int(coords[2]) : int(coords[5]),
            :,
        ]
        prop_np = properties[
            :,
            int(coords[0]) : int(coords[3]),
            int(coords[1]) : int(coords[4]),
            int(coords[2]) : int(coords[5]),
        ]

        field = torch.tensor(field_np, dtype=torch.float32)
        prop = torch.tensor(prop_np, dtype=torch.float32)

        return field, prop
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        field, properties, mask = batch["field"][0], batch["properties"][0], batch["mask"][0]
        field, properties = self._field_and_prop(field, properties, mask)
        field_shift = self._shift_phase(field)
        loss = self.loss_fn.calc_loss(field_shift, properties)
        self.log("loss", loss, prog_bar=True)
        return loss
    

class SingleSampleDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
    
    def __len__(self):
        return 1  # Just one sample
    
    def __getitem__(self, idx):
        return self.data


class TorchOptimizer:

    def __init__(self, cost_function, max_iter: int = 5000) -> None:
        self.cost_fuction = cost_function
        self.max_iter = max_iter

    def optimize(self, simulation: Simulation):
        data = {
            "field": simulation.simulation_raw_data.field,
            "properties": simulation.simulation_raw_data.properties,
            "mask": simulation.simulation_raw_data.subject,
        }

        dataset = SingleSampleDataset(data)
        dataloader = DataLoader(dataset, batch_size=1)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="loss",    # Metric you want to monitor
            mode="min",            # 'min' because lower validation loss is better
            save_top_k=1,          # Save only the best checkpoint
            dirpath="checkpoints", # Directory to store checkpoints
            filename="best_checkpoint"  # Static name; you can also use dynamic formatting.
        )

        trainer = pl.Trainer(
            max_epochs=5000,
            max_time={"minutes": 4},
            accelerator="cpu",
            callbacks=[checkpoint_callback]
        )

        module = Module()

        trainer.fit(module, dataloader)

        best_model = Module.load_from_checkpoint(checkpoint_callback.best_model_path)

        coil_config = CoilConfig(
            phase=best_model.phase.cpu().detach().numpy(),
            amplitude=best_model.amplitude.cpu().detach().numpy(),
        )
        return coil_config
