import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def load(name: str):
    data = pd.read_csv(name)
    x = data.loc[:, data.columns != 'cena'].to_numpy()
    y = data['cena'].to_numpy()

    return x, y

x_train, y_train = load('mieszkania.csv')
x_test, y_test = load('mieszkania_test.csv')

districts = set([row[1] for row in x_train])
sum_meters_dist = {}
sum_price_dist = {}
avg_price = {}
for district in districts:
    sum_meters_dist[district] = 0
    sum_price_dist[district] = 0
n = len(x_train)
for i in range(n):
    sum_price_dist[x_train[i][1]] += y_train[i]
    sum_meters_dist[x_train[i][1]] += x_train[i][0]
for district in districts:
    avg_price[district] = sum_price_dist[district] / sum_meters_dist[district] 
# avg_price to mapa dzielnica -> średnia cena za m^2 w dzielnica

def vectorize(set):
    out = []
    for row in set:
        out.append(np.array([
            row[0] * avg_price[row[1]],
            row[2],
            row[3],
            2023 - row[4],
            row[5]
        ]))
    return np.array(out)

def loss(coefs, data, prices):
    pred = np.dot(data, coefs[1:]) + coefs[0]
    return mean_absolute_percentage_error(prices, pred)

def part_der_local(coefs, data, price, dim):
    if dim == 0:
        scalar = 1
    else:
        scalar = data[dim - 1]
    val = np.dot(data, coefs[1:]) + coefs[0] - price
    numerator = scalar * val
    denumerator = price * abs(val)
    return numerator / denumerator

def part_der(coefs, data, prices, dim):
    n = len(prices)
    results = [part_der_local(coefs, data[i], prices[i], dim) for i in range(n)]
    return np.sum(results) / n

def gradient(coefs, data, prices):
    grad = [part_der(coefs, data, prices, i) for i in range(dim + 1)]
    return np.array(grad)

data = vectorize(x_train)
prices = y_train
dim = len(data[0])
coefs = np.zeros(dim + 1)
lr = 0.005 # learning rate

curr_loss = loss(coefs, data, prices)
prev_loss = curr_loss + 1
while (abs(prev_loss - curr_loss) > 0.000_000_001):
    prev_loss = curr_loss
    coefs = coefs - (lr * gradient(coefs, data, prices))
    curr_loss = loss(coefs, data, prices)
    print(f'''curr_loss={curr_loss}''')
print(f'''coefs: {coefs}''')

print(f'''Loss at train data after training: {curr_loss} ''')

data_test = vectorize(x_test)
prices_test = y_test
print(f'''Loss at test data: {loss(coefs, data_test, prices_test)}''')