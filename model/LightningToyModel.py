from training.base import *
from  pytorch_lightning import LightningModule
from torchmetrics import Accuracy

class LightningToyModel(LightningModule):
    def __init__(self, model, n_classes=10, lr=0.01):
        super().__init__()

        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters() 
        self.model = model
        self.train_loss = 0
        self.val_acc = 0
        self.train_sz = 0
        self.val_sz = 0

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, loss, acc = self.__get_preds_loss_accuracy(batch)
        self.train_loss += loss
        self.train_sz += len(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self.__get_preds_loss_accuracy(batch)
        self.val_acc += acc*len(batch)
        self.val_sz += len(batch)
        return preds
    
    def on_train_epoch_start(self):
        self.train_loss = 0
        self.train_sz = 0

    def on_val_epoch_start(self):
        self.val_acc = 0
        self.val_sz = 0

    def on_train_epoch_end(self):
        self.log('train_loss', self.train_loss/self.train_sz, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc/self.val_sz, prog_bar=True)

    def configure_optimizers(self):
        return  torch.optim.AdamW(self.parameters(), lr=self.lr)

    def __get_preds_loss_accuracy(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = Accuracy('multiclass', num_classes=self.n_classes)(logits.max(1)[1], y)
        return logits.max(1)[1], loss, acc