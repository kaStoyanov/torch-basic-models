import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data


df = pd.read_csv('./lstm-nlp/airline-passengers.csv')
timeseries = df[["Passengers"]].values.astype('float32')

# plt.plot(timeseries)
# plt.show()

print (timeseries.shape)

train_size = int(len(timeseries) * 0.8)
test_size = len(timeseries) - train_size
lookback = 6
train, test = timeseries[:train_size], timeseries[train_size:] 

# print (train.shape, test.shape)

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

x_train, y_train = create_dataset(train, lookback)

print (x_train.shape, y_train.shape)