import torch

x = torch.tensor(1.0)
w = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)


def loss(y , y_pred):
    loss = (y - y_pred)**2
    return loss

def forward(x , w):
    return x * w

def backward(w , loss):
    loss.backward()
    gradient = w.grad
    update(w , gradient , 0.43)
    w.grad.zero_()

def update(w , gradient , learning_rate):
    with torch.no_grad():
        w.data = w - (learning_rate * gradient)

def train(epochs):
    for i in range(epochs):
        print(f"epoch: {i + 1} predicted: {forward(x , w)} loss: {loss( y , forward(x , w) )} w: {w}")
        backward(w , loss( y , forward(x , w) ))


train(10)
