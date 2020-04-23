"""
Contains two models architecture
1) model_scratch

2) model_transfer

Find more about it by printing them
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


use_cuda = torch.cuda.is_available()

class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=3)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)    
        self.pool4 = nn.AdaptiveAvgPool2d((6, 6))
        
        self.fc1 = nn.Linear(256 * 6 * 6, 2048)
        self.fc2 =nn.Linear(2048, 3048)
        self.fc3 = nn.Linear(3048, 4096)
        self.dropout = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(4096, 133)
    def forward(self, x):
        ## Define forward behavior
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_bn(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv5_bn(x)
        x = self.pool3(x)
        x = self.pool4(x)

        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        return x


# Using Pretrained model

model_transfer = models.vgg19_bn(pretrained=False)

#check if cuda is available
if use_cuda:
    model_transfer = model_transfer.cuda()

# freezing
for param in model_transfer.features.parameters():
    param.requires_grad = False


# getting last layer features
n_inputs =  model_transfer.classifier[6].in_features

#Define new linear layer
model_transfer.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs, 1000), 
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(1000, 133))
if use_cuda:
    model_transfer.classifier[6] = nn.Sequential(
                          nn.Linear(n_inputs, 1000), 
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(1000, 133)).cuda()




# Instantiate models
model_scratch = Net()
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()