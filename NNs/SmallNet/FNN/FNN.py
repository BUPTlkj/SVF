import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: 3x8x8 - channel x height x width

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

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

        x = x.view(1, -1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x