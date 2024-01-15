#load data
#model
#optim
#training loop
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def load(path):
    X, Y = path
    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))  # Ensure Y is a 1D tensor
    print(X.shape)
    print(Y.shape)
    return X, Y

# Example usage with make_regression
X, Y = load(datasets.make_regression(n_samples=10, n_features=1, noise=10, random_state=42))
Y = Y.view(-1, 1)

    

class LinearRegression(nn.Module):
    #init model
    #forward pass
    def __init__(self , input_size , output_size):
        super(LinearRegression , self).__init__()
        self.linear = nn.Linear(input_size , output_size)

    def forward(self , X):
        return self.linear(X)

def calc_loss(y , y_pred):
    loss = nn.MSELoss()(y , y_pred)
    return loss

n_samples , n_features  = X.shape
model = LinearRegression(n_features , 1)
optimizer =  optim.SGD(model.parameters() , lr=0.5)

def train(iters):
    
    for epochs in range(iters):
        y_pred = model(X)
        loss = calc_loss(Y , y_pred)
        print(f"epoch: {epochs + 1} loss {loss}")
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()


train(10)



