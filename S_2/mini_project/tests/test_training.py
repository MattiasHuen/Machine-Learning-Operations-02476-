import torch
from torch import Tensor

from src.mlops_mini_project.train import train
from src.mlops_mini_project.model import Model, corrupt_mnist


def test_training_backprop():
    """Test that training performs backpropagation and updates model weights."""
    statistics = train(lr=1e-2, batch_size=64, epochs=1)
    assert statistics["train_loss"][0] > statistics["train_loss"][-1], "Training loss did not decrease"



