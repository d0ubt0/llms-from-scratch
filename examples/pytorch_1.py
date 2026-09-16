from random import random

import torch

N = 100

dim_in = 1
dim_out = 1

X = torch.randn(N, dim_in)

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)

y_true = X @ true_W + true_b + torch.randn(N, dim_in)

W = torch.zeros_like(true_W, requires_grad=True)
b = torch.zeros_like(true_b, requires_grad=True)
print("W", W, "b", b)
EPOCHS = 1000
LR = 0.01

for epoch in range(EPOCHS):
    y_hat = X @ W + b
    loss_mse = ((y_hat - y_true) ** 2).mean()
    loss_mse.backward()
    if epoch % 10 == 0:
        print("W: ", W, "b: ", b, "loss: ", loss_mse)
    with torch.no_grad():
        W -= LR * W.grad
        b -= LR * b.grad

    W.grad.zero_()
    b.grad.zero_()

print("W: ", W, "b: ", b)
