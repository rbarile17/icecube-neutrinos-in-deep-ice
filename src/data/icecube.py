"""IceCube dataset."""

import gc
import random
import pandas as pd

import torch

from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Batch

from .paths import RAW_DATA_PATH, RAW_TRAIN_BATCHES_PATH, RAW_META_PATH
from ..utils import angle_to_xyz


class IceCube(IterableDataset):
    """IceCube dataset."""
    def __init__(
        self,
        batch_ids,
        batch_size=200,
        max_pulses=200,
        shuffle=False,
    ):
        self.batch_ids = batch_ids
        self.batch_size = batch_size
        self.max_pulses = max_pulses
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.batch_ids)

    def handle_distributed(self):
        """Handle num_workers > 1 and multi-gpu."""
        is_dist = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        rank_id = torch.distributed.get_rank() if is_dist else 0

        info = torch.utils.data.get_worker_info()
        num_worker = info.num_workers if info else 1
        worker_id = info.id if info else 0

        num_replica = world_size * num_worker
        offset = rank_id * num_worker + worker_id

        return offset, num_replica

    def __iter__(self):
        offset, num_replica = self.handle_distributed()
        batch_ids = self.batch_ids[offset::num_replica]

        # Sensor data
        sensor_xyz = pd.read_csv(RAW_DATA_PATH / 'sensor_geometry.csv')[['x', 'y', 'z']]
        sensor_xyz = torch.from_numpy(sensor_xyz.values).float()

        # Read each batch and meta iteratively into memory and build mini-batch
        for _, batch_id in enumerate(batch_ids):
            data = pd.read_parquet(RAW_TRAIN_BATCHES_PATH / f'batch_{batch_id}.parquet')
            meta = pd.read_parquet(RAW_META_PATH/ f'meta_{batch_id}.parquet')

            meta = dict(zip(
                meta['event_id'].tolist(),
                angle_to_xyz(torch.from_numpy(meta[['azimuth', 'zenith']].values).float())))

            # Take all event_ids and split them into batches
            event_ids = random.shuffle(list(meta.keys())) if self.shuffle else list(meta.keys())
            event_ids = [
                event_ids[i : i + self.batch_size]
                for i in range(0, len(event_ids), self.batch_size)
            ]

            for event_ids_batch in event_ids:
                batch = []

                # For each sample, extract features
                for event_id in event_ids_batch:
                    event_df = data.loc[event_id]

                    if len(event_df) > self.max_pulses:
                        event_df = event_df.sample(n=self.max_pulses)

                    event_df = event_df.sort_values(['time'])
                    sensor = torch.from_numpy(event_df['sensor_id'].values).long()
                    feat = torch.stack([
                        sensor_xyz[sensor][:, 0],
                        sensor_xyz[sensor][:, 1],
                        sensor_xyz[sensor][:, 2],
                        torch.from_numpy(event_df['time'].values).float(),
                        torch.from_numpy(event_df['charge'].values).float(),
                        torch.from_numpy(event_df['auxiliary'].values).float()], dim=1)

                    batch.append(
                        Data(
                            x=feat,
                            gt=meta[event_id],
                            n_pulses=len(feat),
                            eid=torch.tensor([event_id]).long(),
                        )
                    )

                yield Batch.from_data_list(batch)

            del data
            del meta
            gc.collect()

    def __getitem__(self, idx):
        raise NotImplementedError
