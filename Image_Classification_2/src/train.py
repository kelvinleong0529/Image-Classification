import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
trainset = datasets.SVHN(root='./input', split='train', download=True, transform=transform)
testset = datasets.SVHN(root='./input', split='test', download=True, transform=transform)

# taking a subset from the dataset
# trainset1 = torch.utils.data.Subset(trainset, list(range(0,100)))
# testset1 = torch.utils.data.Subset(testset, list(range(0,100)))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

dataiter = iter(testloader)
images, labels = dataiter.next()

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining 2nd 2D convolution layer
            nn.Conv2d(6,12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining 3rd 2D convolution layer
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining 4th 2D convolution layer
            #nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(48),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining 5th 2D convolution layer
            #nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=1), # first try = 96, second-try = 192
            #nn.BatchNorm2d(192),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(384, 10), # 3-lyr = 384, 4-lyr = 192, 5-lyr = 96
            nn.Softmax(dim=1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

model = Net()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 2)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
hist = []

for i in range(2):
    running_loss = 0
    for images, labels in trainloader:

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
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))
        hist.append(running_loss/len(trainloader))
        scheduler.step(running_loss/len(trainloader)) # for decaying learning rate

# getting predictions on test set and measuring the performance
correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    img = images[i].view(1, 3, 32, 32)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.cpu()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

img, _ = next(iter(trainloader))

# get the most likely prediction of the model
pred = model(img)

# get the gradient of the output with respect to the parameters of the model
pred[:, 386].backward()

# pull the gradients out of the model
gradients = model.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = model.get_activations(img).detach()

# weight the channels by corresponding gradients
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())
plt.show()

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

plt.figure(figsize=(10, 7))
plt.plot(hist, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('E:/Machine Learning/project1/lr=0.01(self,3-lyr,wider,Adam,gamma=5).png')
plt.show()