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

    log_dir = Path('log') / Path(__file__).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    train_set = IceCube(list(range(1, 51)), batch_size=100, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        collate_fn=collate_fn,
    )

    model = DynEdge(num_total_step=num_total_step, num_warmup_step=num_warmup_step)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator='gpu',
        devices=1,
        max_steps=num_total_step,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.ModelCheckpoint(log_dir, save_top_k=-1),
        ],
    )
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()