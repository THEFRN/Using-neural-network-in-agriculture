# Import Libraries
import cv2 as cv
# /////////////////////////////
import matplotlib
import matplotlib.pyplot as plt
# /////////////////////////////
import torch
import torch.nn as nn
# /////////////////////////////
import sys
sys.path.append('usr/local/lib/python3.10.2/site-packages')

# Loading the image data
image = cv.imread('1.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow('ShowImage', image)
cv.waitKey(0)
cv.destroyAllWindows()

batch_size = 64
# Data Preparation
n_input, n_hidden, n_out, batch_size, learning_rate = 10, 30, 1, 100, 0.1

data_x = image
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
matplotlib.pyplot.plot(losses)
matplotlib.pyplot.ylabel('loss')
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.title("Learning rate %f" % learning_rate)
matplotlib.pyplot.show()
