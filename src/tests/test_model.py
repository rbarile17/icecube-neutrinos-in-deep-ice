import pytest
import torch

from src.models import DynEdge
from src.data import IceCube


@pytest.fixture
def model():
    return DynEdge()


@pytest.fixture
def dataset():
    return IceCube([1], batch_size=100)


def test_train_dynedge(model, dataset):
    model = DynEdge()

    losses = []
    for i, example in enumerate(dataset):
        losses.append(model.training_step(example, None, log=False))

        if i == 10:
            break

    assert losses[-1] < losses[0]


@torch.no_grad()
def test_shape(model, dataset):
    dataset_iterator = iter(dataset)
    example = next(dataset_iterator)
    pred = model(example)
    assert pred.shape == (100, 4)


