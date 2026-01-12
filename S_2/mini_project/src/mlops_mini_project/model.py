from pytorch_lightning import LightningModule, Trainer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch import Tensor
# class Model(nn.Module):
#     """My awesome model."""

#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(128, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv3(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         return self.fc1(x)

# src/models/model.py
def forward(self, x: Tensor):
    if x.ndim != 4:
        raise ValueError('Expected input to a 4D tensor')
    if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
        raise ValueError('Expected each sample to have shape [1, 1, 28, 28]')

def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

class Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)
    
    def training_step(self, batch):
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    # def train_dataloader(self):
    #     train_set, _ = corrupt_mnist()
    #     return DataLoader(train_set, batch_size=64, shuffle=True)

    # def test_dataloader(self):
    #     _, test_set = corrupt_mnist()
    #     return DataLoader(test_set, batch_size=64, shuffle=False)
    
train_set, test_set = corrupt_mnist()

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)
                            
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)

if __name__ == "__main__":
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    


    trainer = Trainer(accelerator="cpu", check_val_every_n_epoch=1, max_epochs=10, default_root_dir="../logs", limit_train_batches=0.2, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, test_loader)

    # dummy_input = torch.randn(1, 1, 28, 28)
    # output = model(dummy_input)
    # print(f"Output shape: {output.shape}")
