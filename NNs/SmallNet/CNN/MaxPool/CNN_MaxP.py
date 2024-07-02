import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: 3x8x8 - channel x height x width

class CNN_MaxP(nn.Module):
    def __init__(self):
        super(CNN_MaxP, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=1)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=3)
        self.c3 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=1)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(4, 4)

    def forward(self, x):
        # print('\n','\n')
        # outputs = {}
        # x = F.relu(self.fc1(x))
        # outputs['fc1'] = x
        # x = F.relu(self.fc2(x))
        # outputs['fc2'] = x
        # x = F.relu(self.fc3(x))
        # outputs['fc3'] = x
        # x = self.fc4(x)
        # outputs['fc4'] = x
        # return outputs
        # print("before mat reshape: ", x.shape)
        # x = x.view(1, 192)
        # print("flattened mat reshape: ", x.shape)
        
        x = self.relu(self.c1(x))
        x = self.maxPool(x)
        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))
        x = self.flatten(x)
        x = self.f1(x)
        return x