# Implements the same functionality as LinRegTorch.py but using PyTorch Lightning

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
import numpy as np
import pandas as pd
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Read parameters from parameters.yaml:
learning_rate = 0
epochs = 0
batch_size = 0
with open("parameters.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        batch_size = params['batch_size']
    except yaml.YAMLError as exc:
        print(exc)
        
def get_avg_price(avg_price, X_train, y_train):
    districts = set([row[1] for row in X_train])
    sum_meters_dist = {}
    sum_price_dist = {}
    for district in districts:
        sum_meters_dist[district] = 0
        sum_price_dist[district] = 0
    n = len(X_train)
    for i in range(n):
        sum_price_dist[X_train[i][1]] += y_train[i]
        sum_meters_dist[X_train[i][1]] += X_train[i][0]
    for district in districts:
        avg_price[district] = sum_price_dist[district] / sum_meters_dist[district] 
    # avg_price to mapa dzielnica -> średnia cena za m^2 w dzielnica

def vectorize(set, avg_price):
    # stwierdziłem, że zamiast trzymać metraż kwadratowy, 
    # lepiej trzymać cenę za metr kwadratowy * metraż
    out = []
    for row in set:
        out.append([
            row[0] * avg_price[row[1]],
            row[2],
            row[3],
            2023 - row[4],
            row[5]
        ])
    return torch.tensor(out)

def load(name: str):
        data = pd.read_csv(name)
        x = data.loc[:, data.columns != 'cena'].to_numpy()
        y = data['cena'].to_numpy()
        return x, y
    
class FlatsTrainDataset(Dataset):    
    def __init__(self):
        X_train, y_train= load('mieszkania.csv')
        self.avg_price = {}
        get_avg_price(self.avg_price, X_train, y_train)
        self.X = vectorize(X_train, self.avg_price).float()
        self.y = torch.from_numpy(y_train).float()
                
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
    
    def get_avg_price(self):
        return self.avg_price
    
class FlatsTestDataset(Dataset):
    def __init__(self, avg_price):
        X_test, y_test = load('mieszkania_test.csv')
        self.X = vectorize(X_test, avg_price).float()
        self.y = torch.from_numpy(y_test).float()
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)
    
trainDataset = FlatsTrainDataset()
testDataset = FlatsTestDataset(trainDataset.get_avg_price())

class LinearRegression(pl.LightningModule):
    def __init__(self, num_in, num_out):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_in, num_out)
        self.validation_step_outputs = []
        self.loss_fun = MeanAbsolutePercentageError()
        
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).reshape(-1)
        loss = self.loss_fun(y_hat, y)
        return loss
    
    def train_dataloader(self):
        dataset = trainDataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=learning_rate)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).reshape(-1)
        loss = self.loss_fun(y_hat, y)
        self.validation_step_outputs.append(loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs = []
        # log the avg_loss to tensorboard:
        self.log('val_loss', avg_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        
        
        # print(f'Validation loss: {avg_loss}')

    
    def val_dataloader(self):
        dataset = testDataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
if __name__ == '__main__':
    model = LinearRegression(5,1)
    trainer = Trainer(max_epochs=epochs, log_every_n_steps=13)
    trainer.fit(model)