import torch
import pytest
import os.path
from torch import Tensor

from src.mlops_mini_project.model import Model

# @pytest.mark.parametrize("", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_model_output_shape():
    """Test that the model produces the correct output shape."""
    model = Model() 
    sample_input = torch.randn(1, 1, 28, 28) 
    output = model(sample_input)
    assert output.shape == (1, 10), "Model output shape isn't consistent with the expected shape"


# def test_error_on_wrong_shape():
#     model = Model()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn(1,1,28,28))
#     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 1, 28, 28]'):
#         model(torch.randn(1,1,28,28))

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = Model()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)