"""Evaluate the model on the validation set."""

from pathlib import Path

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .dynedge import collate_fn
from .dynedge import DynEdge
from ..data import IceCube


def main():
    """Main function."""
    pl.seed_everything(123)
    torch.set_float32_matmul_precision('medium')

    # Memory seems to be leaking if not set to 'spawn'
    # Maybe due to https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # It also decrease the memory consumption
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Config
    num_total_step = 100_000
    num_warmup_step = 1_000
    parquet_dir = Path('data/raw/train')
    meta_dir = Path('data/raw/meta')
    assert parquet_dir.exists() and meta_dir.exists()
    log_dir = Path('log') / Path(__file__).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    valid_set = IceCube([51], batch_size=100)
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        collate_fn=collate_fn,
    )

    model = DynEdge(num_total_step=num_total_step, num_warmup_step=num_warmup_step)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator='gpu',
        devices=1,
        max_steps=num_total_step,
    )

    # Verify with offical pretrained weight
    weights = torch.load('./models/state_dict.pth')
    new_weights = {}
    for key, value in weights.items():
        key = key.replace('_gnn._conv_layers.0', 'conv0')
        key = key.replace('_gnn._conv_layers.1', 'conv1')
        key = key.replace('_gnn._conv_layers.2', 'conv2')
        key = key.replace('_gnn._conv_layers.3', 'conv3')
        key = key.replace('_gnn._post_processing', 'post')
        key = key.replace('_gnn._readout', 'readout')
        key = key.replace('_tasks.0._affine', 'pred')
        new_weights[key] = value
    print(model.load_state_dict(new_weights))
    print(trainer.validate(model, valid_loader))

if __name__ == '__main__':
    main()
