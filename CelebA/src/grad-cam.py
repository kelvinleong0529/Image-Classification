import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms

# load the model
model = models.resnet50(pretrained=True)
model.eval()
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
relu_layer = [] # for activation layer
# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
                elif type(child) == nn.ReLU:
                    relu_layer.append(child)
print(f"Total convolutional layers: {counter}")

# read and visualize an image
img = cv.imread("E:/Machine Learning/project2/input/face-classifier/Images/000001.png")
img2 = img
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
# define the transforms
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((512, 512)),transforms.ToTensor(),])
img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results

# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    if num_layer == 5:
        break
    plt.figure(figsize=(30, 30))
    print(outputs[num_layer].size())
    layer_viz = outputs[num_layer][0, :, :, :]
    print(model_weights[num_layer].size())
    layer_viz *= relu_layer[num_layer]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    heatmap = torch.mean(layer_viz,dim=1).squeeze()
    heatmap = np.maximum(heatmap,0)
    heatmap /= torch.max(heatmap,0)
    plt.imshow(heatmap)
    plt.show()
    bz,nc,h,w = model_weights[num_layer].shape
    print(bz,' ',nc,' ',h,' ',w)
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        beforeDot = model_weights[num_layer].reshape((nc,h*w))
        print(beforeDot)
        plt.imshow()
        #plt.imshow(filter, cmap='gray')
        
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"E:/Machine Learning/project2/outputs/layer_{num_layer}.png")
    # plt.show()
    plt.close()