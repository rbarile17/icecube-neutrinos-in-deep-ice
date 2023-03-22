import pytest

from torch_geometric.data import Data, Batch

from src.data import IceCube

BATCH_SIZE = 100

@pytest.fixture
def dataset():
    return IceCube([1], batch_size=BATCH_SIZE)


def test_shape(dataset):
    example = next(iter(dataset))

    assert example.x.shape == (example.n_pulses.sum().item(), 6)
    assert example.gt.shape[0] == BATCH_SIZE * 3    


    