import pyarrow
import pyarrow.parquet

import pandas as pd

from tqdm import tqdm

from .paths import RAW_DATA_PATH, PROCESSED_META_PATH

out_dir = PROCESSED_META_PATH
out_dir.mkdir(parents=True, exist_ok=True)

meta = pyarrow.parquet.read_table(RAW_DATA_PATH / 'train_meta.parquet')
for batch_id in tqdm(range(1, 661)):
    group = meta.filter(pyarrow.compute.field('batch_id') == batch_id)
    group = group.select(['event_id', 'azimuth', 'zenith'])
    pyarrow.parquet.write_table(group, out_dir / f'meta_{batch_id}.parquet')

df = pd.read_parquet(RAW_DATA_PATH / 'test_meta.parquet')
df['azimuth'] = 0.0
df['zenith'] = 0.0
df.to_parquet(out_dir / 'meta_661.parquet')