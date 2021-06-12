import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader

#from torch.optim import SGD
#from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.style.use('ggplot')

# initialize the computation device5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = models.model(pretrained=True, requires_grad=False).to(device)
# learning parameters
lr = 0.01
epochs = 30
batch_size = 15
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(),lr = lr)
criterion = nn.BCELoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.01)

# read the training csv file
train_csv = pd.read_csv("E:/Machine Learning/project2/input/face-classifier/Mutli_Label_dataset/list_attr_celeba.txt",sep =" ",nrows=1000)
# train dataset
train_data = ImageDataset(train_csv, train=True, test=False)
# validation dataset
valid_data = ImageDataset(train_csv, train=False, test=False)
# train data loader
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
# validation data loader
valid_loader = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
    valid_epoch_loss = validate(model, valid_loader, criterion, valid_data, device)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')
    # scheduler.step(valid_epoch_loss/len(valid_loader)) # for decaying learning rate
    # save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, 'E:/Machine Learning/project2/outputs/model.pth')
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('E:/Machine Learning/project2/outputs/loss2.png')
plt.show()