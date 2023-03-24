"""Implementation of DynEdge model."""

import torch
import numpy as np
from torch import nn

import pytorch_lightning as pl
from torch_geometric.nn import knn_graph, EdgeConv
from torch_geometric.utils import homophily
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_max, scatter_min

from ..utils import VonMisesFisher2DLoss
from ..utils import angular_error, eps_like


class MLP(nn.Sequential):
    """Implementation of a simple MLP."""
    def __init__(self, feats):
        layers = []
        for i in range(1, len(feats)):
            layers.append(nn.Linear(feats[i - 1], feats[i]))
            layers.append(nn.LeakyReLU())
        super().__init__(*layers)


class DynEdge2DLoss(pl.LightningModule):
    """Implementation of DynEdge model."""

    # pylint: disable=unused-argument
    def __init__(
        self, max_lr=1e-3, min_lr=1e-5, num_warmup_step=1_000, num_total_step=20_000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv0 = EdgeConv(MLP([34, 128, 256]), aggr='add')
        self.conv1 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.conv2 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.conv3 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.post = MLP([1041, 336, 256])
        self.readout = MLP([768, 128])
        self.pred = nn.Linear(128, 2)

    # pylint: disable=arguments-differ
    def forward(self, data: Batch):
        """Forward pass of the model."""
        vert_feat = data.x

        vert_feat[:, 0] /= 500.0  # x
        vert_feat[:, 1] /= 500.0  # y
        vert_feat[:, 2] /= 500.0  # z
        vert_feat[:, 3] = (vert_feat[:, 3] - 1.0e04) / 3.0e4  # time
        vert_feat[:, 4] = torch.log10(vert_feat[:, 4]) / 3.0  # charge

        edge_index = knn_graph(vert_feat[:, :3], 8, data.batch)

        # Construct global features
        global_feats = torch.cat([
            scatter_mean(vert_feat, data.batch, dim=0),
            homophily(edge_index, vert_feat[:, 0], data.batch).reshape(-1, 1),
            homophily(edge_index, vert_feat[:, 1], data.batch).reshape(-1, 1),
            homophily(edge_index, vert_feat[:, 2], data.batch).reshape(-1, 1),
            homophily(edge_index, vert_feat[:, 3], data.batch).reshape(-1, 1),
            torch.log10(data.n_pulses).reshape(-1, 1)
        ], dim=1)  # [B, 11]

        # Distribute global_feats to each vertex
        _, cnts = torch.unique_consecutive(data.batch, return_counts=True)
        global_feats = torch.repeat_interleave(global_feats, cnts, dim=0)
        vert_feat = torch.cat((vert_feat, global_feats), dim=1)

        # Convolutions
        feats = [vert_feat]
        # Conv 0
        vert_feat = self.conv0(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 1
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=data.batch)
        vert_feat = self.conv1(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 2
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=data.batch)
        vert_feat = self.conv2(vert_feat, edge_index)
        feats.append(vert_feat)
        # Conv 3
        edge_index = knn_graph(vert_feat[:, :3], k=8, batch=data.batch)
        vert_feat = self.conv3(vert_feat, edge_index)
        feats.append(vert_feat)

        # Postprocessing
        post_inp = torch.cat(feats, dim=1)
        post_out = self.post(post_inp)

        # Readout
        readout_inp = torch.cat(
            [
                scatter_min(post_out, data.batch, dim=0)[0],
                scatter_max(post_out, data.batch, dim=0)[0],
                scatter_mean(post_out, data.batch, dim=0),
            ],
            dim=1,
        )
        readout_out = self.readout(readout_inp)

        # Predict
        pred = self.pred(readout_out)
        kappa = torch.linalg.vector_norm(pred, dim=1) + eps_like(pred)
        azimuth = torch.atan2(pred[:, 1], pred[:, 0])
        azimuth = torch.where(
            azimuth < 0, azimuth + 2 * np.pi, azimuth
        )
        pred = torch.stack((azimuth, kappa), dim=1)

        return pred
    

    def train_or_valid_step(self, data, prefix, log=True):
        """Training and validation step."""
        pred_azimuth_k = self.forward(data)  # [B, 2]
        true_azimuth = data.gt_angle.view(-1, 1)  # [B, 1]
        loss = VonMisesFisher2DLoss()(pred_azimuth_k, true_azimuth)

        # error = angular_error(pred_xyzk[:, :3], true_xyz).mean()

        if log:
            self.log(f'loss/{prefix}', loss, batch_size=len(true_azimuth))
            # self.log(f'error/{prefix}', error, batch_size=len(true_azimuth))
        return loss

    def training_step(self, data, _, log=True):
        return self.train_or_valid_step(data, 'train', log)

    def validation_step(self, data, _, log=True):
        self.train_or_valid_step(data, 'valid', log)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.max_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, 1e-12, 1.0, self.hparams.num_warmup_step
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.hparams.num_total_step, self.hparams.min_lr
                ),
            ],
            milestones=[self.hparams.num_warmup_step],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }


def collate_fn(data):
    """Collate function for PyTorch DataLoader."""
    return data[0]
