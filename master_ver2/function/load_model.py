import torch
import torch.nn as nn
import pytorch_lightning as pl


class Individual_RegressionModel(pl.LightningModule):
    def __init__(self):
        super(Individual_RegressionModel, self).__init__()
        self.model = nn.Sequential(
                      nn.Linear(7, 128),
                      nn.BatchNorm1d(128),
                      nn.Dropout(0.5),
                      nn.Linear(128,64),
                      nn.LeakyReLU(0.1 , inplace = True),
                      nn.BatchNorm1d(64),
                      nn.Dropout(0.5),
                      nn.Linear(64,32),
                      nn.LeakyReLU(0.1, inplace = True),
                      nn.BatchNorm1d(32),
                      nn.Linear(32,2)
        ) 
        
        
    def forward(self, x):
        
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    
