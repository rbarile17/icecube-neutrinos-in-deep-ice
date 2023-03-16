import gc
import random
import pandas as pd

import torch

from torch.utils.data import IterableDataset
from torch_geometric.data import Data, Batch

from .paths import RAW_DATA_PATH, RAW_TRAIN_BATCHES_PATH, RAW_META_PATH
from ..utils import angle_to_xyz


class IceCube(IterableDataset):
    def __init__(
        self,
        chunk_ids,
        batch_size=200,
        max_pulses=200,
        shuffle=False,
    ):
        self.chunk_ids = chunk_ids
        self.batch_size = batch_size
        self.max_pulses = max_pulses
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.chunk_ids)

    def __iter__(self):
        # Handle num_workers > 1 and multi-gpu
        is_dist = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_dist else 1
        rank_id = torch.distributed.get_rank() if is_dist else 0

        info = torch.utils.data.get_worker_info()
        num_worker = info.num_workers if info else 1
        worker_id = info.id if info else 0

        num_replica = world_size * num_worker
        offset = rank_id * num_worker + worker_id
        chunk_ids = self.chunk_ids[offset::num_replica]

        # Sensor data
        sensor_xyz = pd.read_csv(RAW_DATA_PATH / 'sensor_geometry.csv')[['x', 'y', 'z']]
        sensor_xyz = torch.from_numpy(sensor_xyz.values).float()

        # Read each chunk and meta iteratively into memory and build mini-batch
        for c, chunk_id in enumerate(chunk_ids):
            data = pd.read_parquet(RAW_TRAIN_BATCHES_PATH / f'batch_{chunk_id}.parquet')

            meta = pd.read_parquet(RAW_META_PATH/ f'meta_{chunk_id}.parquet')
            angles = meta[['azimuth', 'zenith']].values
            angles = torch.from_numpy(angles).float()
            xyzs = angle_to_xyz(angles)
            meta = {eid: xyz for eid, xyz in zip(meta['event_id'].tolist(), xyzs)}

            # Take all eventi_ids and split them into batches
            eids = list(meta.keys())
            if self.shuffle:
                random.shuffle(eids)
            eids_batches = [
                eids[i : i + self.batch_size]
                for i in range(0, len(eids), self.batch_size)
            ]

            for batch_eids in eids_batches:
                batch = []

                # For each sample, extract features
                for eid in batch_eids:
                    df = data.loc[eid]
                    if len(df) > self.max_pulses:
                        df = df.sample(n=self.max_pulses)
                    df = df.sort_values(['time'])
                    t = torch.from_numpy(df['time'].values).float()
                    c = torch.from_numpy(df['charge'].values).float()
                    s = torch.from_numpy(df['sensor_id'].values).long()
                    p = sensor_xyz[s]
                    a = torch.from_numpy(df['auxiliary'].values).float()
                    feat = torch.stack([p[:, 0], p[:, 1], p[:, 2], t, c, a], dim=1)

                    batch.append(
                        Data(
                            x=feat,
                            gt=meta[eid],
                            n_pulses=len(feat),
                            eid=torch.tensor([eid]).long(),
                        )
                    )

                yield Batch.from_data_list(batch)

            del data
            del meta
            gc.collect()
