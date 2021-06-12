import torch
import torch.nn as nn
import models
from torchvision.models import resnet152
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ResNet Class
class ResNet(nn.Module):
    def __init__(self,i):
        super(ResNet, self).__init__()
        
        # load the pretrained resnet
        self.resnet = resnet152(pretrained=True)

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=i, stride=3, padding=0, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        
        # average pooling layer
        self.avgpool = self.resnet.avgpool
        
        # classifier
        self.classifier = self.resnet.fc
        
        # gradient placeholder
        self.gradient = None
    
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forward(self, x):
        
        # extract the features
        x = self.features(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x

      
# all the data transformation and loading
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = ImageFolder('E:/Machine Learning/project2/Data', transform=transform)
dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

plt.figure(figsize=(150, 150))

count = 1

for j in range(2,20,1):
    # init the resnet
    resnet = ResNet(j)

    # set the evaluation mode
    _ = resnet.eval()

    # get the image
    img, _ = next(iter(dataloader))

    # forward pass
    pred = resnet(img)

    pred.argmax(dim=1)  # prints tensor([2])

    # get the gradient of the output with respect to the parameters of the model
    pred[:, 32].backward()

    # pull the gradients out of the model
    gradients = resnet.get_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = resnet.get_activations(img).detach()

    # weight the channels by corresponding gradients
    #for i in range(512):
    for i in range(128):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.squeeze())

    # make the heatmap to be a numpy array
    heatmap = heatmap.numpy()

    # interpolate the heatmap
    img = cv2.imread('E:/Machine Learning/project2/Data/CelebA/000004.png')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.7, 0)
    plt.imshow(superimposed_img)
    plt.axis("off")
    count += 1
    plt.savefig(f"E:/Machine Learning/project2/resnet152/resnetCk={j}.png")
    plt.close()
