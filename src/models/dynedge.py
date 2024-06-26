"""Implementation of DynEdge model."""

import torch

import numpy as np

import pytorch_lightning as pl

from abc import abstractmethod

from torch import nn
from torch_geometric.nn import knn_graph, EdgeConv
from torch_geometric.utils import homophily
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_scatter import scatter_mean, scatter_max, scatter_min

from ..utils import VonMisesFisher2DLoss, VonMisesFisher3DLoss
from ..utils import eps_like, angle_to_xyz, angular_error


class MLP(nn.Sequential):
    """Implementation of a simple MLP."""
    def __init__(self, feats):
        layers = []
        for i in range(1, len(feats)):
            layers.append(nn.Linear(feats[i - 1], feats[i]))
            layers.append(nn.LeakyReLU())
        super().__init__(*layers)


class DynEdge(pl.LightningModule):
    """Implementation of DynEdge model."""

    # pylint: disable=unused-argument
    def __init__(
        self, 
        max_lr=1e-3, min_lr=1e-5, 
        num_warmup_step=1_000, num_total_step=20_000
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=17, 
            hidden_size=128,
            num_layers=4,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.5)

        self.conv0 = EdgeConv(MLP([34, 128, 256]), aggr='add')
        self.conv1 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.conv2 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.conv3 = EdgeConv(MLP([512, 336, 256]), aggr='add')
        self.post = MLP([1041, 336, 256])
        self.readout_gnn = MLP([768, 256])
        self.readout = MLP([512, 128])

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
        _, counts = torch.unique_consecutive(data.batch, return_counts=True)
        global_feats = torch.repeat_interleave(global_feats, counts, dim=0)
        vert_feat = torch.cat((vert_feat, global_feats), dim=1)

        vert_feat_lstm = torch.split(vert_feat, counts.tolist(), dim=0)
        vert_feat_lstm = pad_sequence(vert_feat_lstm, batch_first=True)
        vert_feat_lstm = pack_padded_sequence(vert_feat_lstm, counts.tolist(), batch_first=True, enforce_sorted=False)

        packed_output = self.lstm(vert_feat_lstm)[0]
        tuple_output = torch.split(packed_output.data, counts.tolist(), dim=0)

        lstm_forward_out = torch.stack([event[-1][:128] for event in tuple_output])
        lstm_reverse_out = torch.stack([event[0][128:] for event in tuple_output])
        lstm_out = torch.cat((lstm_forward_out, lstm_reverse_out), 1)
        lstm_out = self.drop(lstm_out)

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
        readout_gnn_inp = torch.cat(
            [
                scatter_min(post_out, data.batch, dim=0)[0],
                scatter_max(post_out, data.batch, dim=0)[0],
                scatter_mean(post_out, data.batch, dim=0),
            ],
            dim=1,
        )
        readout_gnn_out = self.readout_gnn(readout_gnn_inp)
        readout_inp = torch.cat([readout_gnn_out, lstm_out], dim=1)
        readout_out = self.readout(readout_inp)

        return readout_out


    @abstractmethod
    def train_or_valid_step(self, data, prefix, log=True):
        """Training and validation step."""
        raise NotImplementedError


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
                )
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
    

class DynEdge3DLoss(DynEdge):
    """Implementation of DynEdge model."""

    # pylint: disable=unused-argument
    def __init__(
        self, 
        max_lr=1e-3, min_lr=1e-5, 
        num_warmup_step=1_000, num_total_step=20_000
    ):
        super().__init__(max_lr, min_lr, num_warmup_step, num_total_step)
        self.pred = nn.Linear(128, 3)

    # pylint: disable=arguments-differ
    def forward(self, data: Batch):
        """Forward pass of the model."""
        readout_out = super().forward(data)
    
        pred = self.pred(readout_out)
        kappa = pred.norm(dim=1, p=2) + 1e-8
        pred_x = pred[:, 0] / kappa
        pred_y = pred[:, 1] / kappa
        pred_z = pred[:, 2] / kappa
        pred = torch.stack([pred_x, pred_y, pred_z, kappa], dim=1)

        return pred


    def train_or_valid_step(self, data, prefix, log=True):
        """See base class."""
        pred_xyz = self.forward(data)  # [B, 4]
        true_xyz = data.gt.view(-1, 3)  # [B, 3]

        loss = VonMisesFisher3DLoss()(pred_xyz, true_xyz).mean()
        error = angular_error(pred_xyz[:, :3], true_xyz).mean()

        if log:
            self.log(f'loss/{prefix}', loss, batch_size=len(true_xyz))
            self.log(f'error/{prefix}', error, batch_size=len(true_xyz))
        return loss


class DynEdge2DLoss(DynEdge):
    """Implementation of DynEdge model."""

    # pylint: disable=unused-argument
    def __init__(
        self, max_lr=1e-3, min_lr=1e-5, num_warmup_step=1_000, num_total_step=20_000
    ):
        super().__init__(max_lr, min_lr, num_warmup_step, num_total_step)
        self.pred_azimuth_layer = nn.Linear(128, 2)
        self.pred_zenith_layer = nn.Linear(128, 2)


    # pylint: disable=arguments-differ
    def forward(self, data: Batch):
        """Forward pass of the model."""
        readout_out = super().forward(data)

        pred_azimuth = self.pred_azimuth_layer(readout_out)
        kappa = torch.linalg.vector_norm(pred_azimuth, dim=1) + eps_like(pred_azimuth)
        azimuth = torch.atan2(pred_azimuth[:, 1], pred_azimuth[:, 0])
        azimuth = torch.where(
            azimuth < 0, azimuth + 2 * np.pi, azimuth
        )
        pred_azimuth_k = torch.stack((azimuth, kappa), dim=1)

        pred_zenith = self.pred_zenith_layer(readout_out)
        zenith = torch.sigmoid(pred_zenith[:, 0]) * np.pi
        kappa = torch.abs(pred_zenith[:, 1]) + eps_like(pred_zenith)

        pred_zenith_k = torch.stack((zenith, kappa), dim=1)

        return pred_azimuth_k, pred_zenith_k
    

    def train_or_valid_step(self, data, prefix, log=True):
        """Training and validation step."""
        pred_azimuth_k, pred_zenith_k = self.forward(data)  # [B, 2]
        true_angles = data.gt_angle.view(-1, 2)  # [B, 2]
        true_azimuth, true_zenith = true_angles[:, 0].unsqueeze(dim=1), true_angles[:, 1].unsqueeze(dim=1)
        loss_azimuth = VonMisesFisher2DLoss()(pred_azimuth_k, true_azimuth).mean()
        loss_zenith = VonMisesFisher2DLoss()(pred_zenith_k, true_zenith).mean()

        pred_xyz = angle_to_xyz(torch.stack([pred_azimuth_k[:, 0], pred_zenith_k[:, 0]], dim=1))
        true_xyz = data.gt.view(-1, 3)

        loss = loss_azimuth + loss_zenith
        error = angular_error(pred_xyz, true_xyz).mean()

        if log:
            self.log(f'loss/{prefix}', loss, batch_size=len(true_angles))
            self.log(f'error/{prefix}', error, batch_size=len(true_azimuth))
        return loss

