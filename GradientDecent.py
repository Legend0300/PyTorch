import torch

x = torch.tensor(1.0)
w = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)


def loss(y , y_pred):
    loss = ((y - y_pred)**2).mean()
    return loss

def forward(x , w):
    return x * w

def backward(w , y , y_pred):
    calc_loss = loss(y , y_pred)
    calc_loss.backward()
    gradient = w.grad
    update(w , gradient , 0.43)
    w.grad.zero_()

def update(w , gradient , learning_rate):
    with torch.no_grad():
        w -= (learning_rate * gradient)

def train(epochs):
    for i in range(epochs):
        y_pred = forward(x , w)
        print(f"epoch: {i + 1} predicted: {y_pred} loss: {loss( y , y_pred )} w: {w}")
        backward(w , y , y_pred)


train(10)
