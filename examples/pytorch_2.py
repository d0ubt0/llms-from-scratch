import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)


dim_in = 5
dim_out = 1

X = torch.randn(100, dim_in)

W_real = torch.randn(dim_in, 1)
b_real = torch.randn(1)

y_real = X @ W_real + b_real + torch.randn(100, 1) * 0.1

print(y_real)

linear = LinearRegression(dim_in, dim_out)

EPOCHS = 1000

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
for epoch in range(EPOCHS):
    y_pred = linear(X)
    loss = torch.nn.functional.mse_loss(y_pred, y_real)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
