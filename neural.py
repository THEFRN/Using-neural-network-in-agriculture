# Import Libraries
import cv2 as cv

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import os
# Adding images
path = 'E:\\Neural Network Project\\First_project\\images'
image_format = '.jpg'
image_file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(image_format)]
# Loading the image data
images = []
for img in image_file_list:
    print(img)
    image = cv.imread(img)
    images.append(image)
    cv.imshow('Image', image)
    cv.waitKey(0)


# Data Preparation
n_hidden, n_out, learning_rate, data_y = 30, 1, 0.1, 0
for image in images:
    data_x = image
    data_y += 1


# Defining Neural Network Model

model = nn.Sequential(nn.Linear(16, n_hidden),
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
