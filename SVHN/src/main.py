import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms
import cv2 as cv
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

data = scipy.io.loadmat('./input/train_32x32.mat')
test_data = scipy.io.loadmat('./input/test_32x32.mat')

# load the train and test set
x_train = data["X"]
y_train = data["y"]
x_test = test_data["X"]
y_test = test_data["y"] 

x_train = np.moveaxis(x_train, -1, 0) # restructure the axis of the train image set
x_test = np.moveaxis(x_test, -1, 0) # restructure the axis of the test image set

x_train = x_train.astype('float64') # convert the train image type to float64
y_train = y_train.astype('int64')   # convert the train label type to int64
x_test = x_test.astype('float64') # convert the test image type to float64
y_test = y_test.astype('int64')   # convert the test label type to int64

x_train /= 255.0    # normalize the train image set
x_test /= 255.0    # normalize the train image set

lb = LabelBinarizer() # encoding for train labels
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.15, random_state=22)

class Net(nn.Module):   
  def __init__(self):
      super(Net, self).__init__()

      self.cnn_layers = nn.Sequential(
          # Defining a 2D convolution layer
          nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(6),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          # Defining another 2D convolution layer
          nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(6),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
      )

      self.linear_layers = nn.Sequential(
          nn.Linear(4 * 7 * 7, 10)
      )

  # Defining the forward pass    
  def forward(self, x):
      x = self.cnn_layers(x)
      x = x.view(x.size(0), -1)
      x = self.linear_layers(x)
      return x

model = Net()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)

for i in range(10):
    running_loss = 0
    for images, labels in data:

        if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()

        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(data)))