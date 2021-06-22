import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms,models
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

def model(pretrained, requires_grad):
    model = models.resnet18(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 10 classes in total
    model.fc = nn.Linear(512, 10) #for resnet18 & resnet34
    # model.fc = nn.Linear(2048,10) 
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = model(pretrained=True, requires_grad=False).to(device)

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
hist = []

for i in range(30):
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


plt.figure(figsize=(10, 7))
plt.plot(hist, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('E:/Machine Learning/project1/lr=0.001(ResNet18,Adam).png')
plt.show()
