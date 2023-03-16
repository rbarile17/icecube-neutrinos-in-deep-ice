pip install -U \
    torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_scatter torch_sparse torch_cluster torch_geometric \
    -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pytorch-lightning tensorboard
pip install ipykernel ipython nbformat tabulate
pip install matplotlib plotly
pip install pandas pyarrow
pip install dask dask-ml