# Import Libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Data Preparation
n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.01

data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()

print(data_x.size())
print(data_y.size())

# Defining Neural Network Model

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())
print(model)
# Loss function and optimizer

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop

losses = []
for epoch in range(20000):
    pred_y = model(data_x)
    loss = loss_function(pred_y, data_y)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()

# Output
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f" % learning_rate)
plt.show()
