import torch

from src.mlops_mini_project.model import Model

def test_model_output_shape():
    """Test that the model produces the correct output shape."""
    model = Model() 
    sample_input = torch.randn(1, 1, 28, 28) 
    output = model(sample_input)
    assert output.shape == (1, 10) 