from torch.utils.data import Dataset
from src.mlops_mini_project.data import MyDataset, corrupt_mnist
from tests import _PATH_DATA



def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(_PATH_DATA + "/raw")
    assert isinstance(dataset, Dataset)

def test_data():
    train_set, test_set = corrupt_mnist()
    assert len(train_set) == 30000  # N_train for training
    assert len(test_set) == 5000   # N_test for test
    assert train_set[0][0].shape == (1, 28, 28) or train_set[0][0].shape == (784,)
    assert test_set[0][0].shape == (1, 28, 28) or test_set[0][0].shape == (784,)
    assert all(label in [i for i in range(10)] for label in [target for _, target in train_set])
    assert all(label in [i for i in range(10)] for label in [target for _, target in test_set])