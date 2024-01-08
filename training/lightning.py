from training.base import *
from model.ToyModel import ToyModel
from  pytorch_lightning import Trainer
from training.base import base_setup_data_loaders
from model import LightningToyModel

transposition = np.arange(60000)
np.random.seed(42)
np.random.shuffle(transposition)   
model = LightningToyModel(ToyModel(28*28, 10))
train_loader, val_loader = base_setup_data_loaders(transposition)

trainer = Trainer(
    accelerator="cpu",
    devices=2,
    strategy="ddp",
    max_epochs=15)

trainer.fit(model, train_loader, val_loader)