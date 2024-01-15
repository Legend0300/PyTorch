import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
X, Y = data.data, data.target

# Skip the view operation for Y

# scaling:
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(-1, 1)


# Model:
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        value = torch.sigmoid(self.linear(X))
        return value

n_samples, n_features = X.shape
model = LogisticRegression(n_features, 1)

optimizer = optim.SGD(model.parameters(), lr=0.3)

def train(n_iters):
    for epoch in range(n_iters):
        y_pred = model(X_train)  # Convert X to a PyTorch tensor
        loss = nn.BCELoss()(y_pred, y_train)

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

train(10)