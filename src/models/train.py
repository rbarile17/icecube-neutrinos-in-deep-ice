"""Evaluate the model on the validation set."""

from pathlib import Path

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .dynedge import DynEdge3DLoss, DynEdge2DLoss
from ..data import IceCube

NUM_EVENTS = 10_000_000
MAX_EPOCHS = 30
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1_024
ACCUMULATE_GRAD_BATCHES = 4

def main():
    """Main function."""
    pl.seed_everything(123)
    torch.set_float32_matmul_precision('medium')

    # Memory seems to be leaking if not set to 'spawn'
    # Maybe due to https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # It also decrease the memory consumption
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Config
    num_total_step = (NUM_EVENTS / (TRAIN_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES)) * MAX_EPOCHS
    num_warmup_step = (NUM_EVENTS / (TRAIN_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES)) / 2

    log_dir = Path('log') / Path(__file__).stem
    log_dir.mkdir(parents=True, exist_ok=True)

    train_set = IceCube(list(range(1, 51)), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=lambda data: data[0])

    valid_set = IceCube([51], batch_size=EVAL_BATCH_SIZE)
    valid_loader = DataLoader(valid_set, batch_size=1, collate_fn=lambda data: data[0])

    model = DynEdge2DLoss(num_total_step=num_total_step, num_warmup_step=num_warmup_step)
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator='gpu',
        devices=1,
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(monitor="loss/valid", mode="min", patience=5),
            pl.callbacks.ModelCheckpoint(log_dir, save_top_k=-1),
            pl.callbacks.RichProgressBar(),
        ],
    )
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()
