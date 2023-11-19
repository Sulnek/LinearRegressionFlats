import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
import numpy as np
import pandas as pd
import yaml

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
class FlatsDataset(Dataset):    
    def __init__(self):
        X_train, y_train= load('mieszkania.csv')
        self.avg_price = {}
        get_avg_price(self.avg_price, X_train, y_train)
        self.X = vectorize(X_train, self.avg_price).float().to(device)
        self.y = torch.from_numpy(y_train).float().to(device)
                
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
    
class LinearRegression(nn.Module):
    def __init__(self, dataset):
        super(LinearRegression, self).__init__()
        input, output = dataset[0]
        self.linear = nn.Linear(list(input.size())[0], 1)
    def forward(self, x):
        return self.linear(x)
        

# Note: Podział na batche jest kompletnie niepotrzebny, ale chciałem go przećwiczyć
dataset = FlatsDataset()
model = LinearRegression(dataset).to(device)
loss = MeanAbsolutePercentageError()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0)

# Training loop:
for i in range(epochs):
    for j, (X, y) in enumerate(loader):    
        # Forward pass
        y_pred = model(X).reshape(-1)
        l = loss(y_pred, y)
        # Backward pass
        l.backward()
        # Update parameters
        optimizer.step()
        # Zero gradients
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f'''i: {i}, loss: {l.item()}''')
        
# Test loop:
X_test, y_test = load('mieszkania_test.csv')
X_test = vectorize(X_test, dataset.avg_price).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
y_pred = model(X_test).reshape(-1)
l = loss(y_pred, y_test)
print(f'''Test loss: {l.item()}''')

# Test zapisu i odczytu modelu:
FILE = "model.pth"
torch.save(model.state_dict(), FILE)

model = None

model = LinearRegression(dataset=dataset).to(device)
model.load_state_dict(torch.load(FILE))
y_pred = model(X_test).reshape(-1)
l = loss(y_pred, y_test)
print(f'''Test loss loaded model: {l.item()}''')
