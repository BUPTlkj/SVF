import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class CNN_No_MaxP(nn.Module):
    def __init__(self):
        super(CNN_No_MaxP, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=1)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(4, 4)

    def forward(self, x):
        print('\n','\n')
        outputs = {}

        x = self.c1(x)
        outputs['conv1'] = x
        x = self.relu(x)
        outputs['relu1'] = x
        x = self.c2(x)
        outputs['conv2'] = x
        x = self.relu(x)
        outputs['relu2'] = x
        x = self.c3(x)
        outputs['conv3'] = x
        x = self.relu(x)
        outputs['relu3'] = x
        x = self.flatten(x)
        outputs['flatten'] = x
        x = self.f1(x)
        outputs['fc1'] = x

        return outputs


# 数据输入
data = [
    0.1523, 0.6152, 0.7813, 0.2013, 0.4657, 0.3122, 0.9834, 0.6745,
    0.4565, 0.1239, 0.8765, 0.7621, 0.2109, 0.5402, 0.1087, 0.9283,
    0.6827, 0.5734, 0.1908, 0.0356, 0.8964, 0.7653, 0.2109, 0.4168,
    0.9572, 0.4853, 0.7319, 0.9157, 0.1712, 0.9935, 0.5278, 0.2920,
    0.9382, 0.1763, 0.5117, 0.6216, 0.4518, 0.1380, 0.8712, 0.3928,
    0.8491, 0.2759, 0.5298, 0.1321, 0.7914, 0.6307, 0.4059, 0.7940,
    0.6555, 0.3316, 0.1270, 0.4833, 0.6741, 0.8573, 0.9946, 0.1382,
    0.2987, 0.6326, 0.9638, 0.3447, 0.1329, 0.8796, 0.9237, 0.6239,

    0.7416, 0.5454, 0.0782, 0.2673, 0.3637, 0.1589, 0.1397, 0.3935,
    0.9339, 0.4017, 0.4629, 0.1104, 0.9167, 0.7280, 0.5240, 0.4681,
    0.6623, 0.1465, 0.0519, 0.8317, 0.9636, 0.4003, 0.7753, 0.7506,
    0.8249, 0.6362, 0.1959, 0.9026, 0.7696, 0.2706, 0.3796, 0.5390,
    0.2539, 0.9583, 0.8753, 0.3819, 0.6211, 0.3104, 0.0586, 0.1519,
    0.1338, 0.4104, 0.4034, 0.6226, 0.6871, 0.2581, 0.5601, 0.9425,
    0.9761, 0.8184, 0.0824, 0.3965, 0.7115, 0.5365, 0.3036, 0.8442,
    0.7255, 0.8164, 0.9452, 0.2346, 0.9093, 0.8230, 0.6509, 0.9971,
    
    0.4352, 0.4629, 0.9906, 0.1272, 0.3095, 0.8240, 0.5988, 0.8124,
    0.2109, 0.5469, 0.7701, 0.4873, 0.1907, 0.3450, 0.3141, 0.6903,
    0.6390, 0.8795, 0.7878, 0.8185, 0.5318, 0.9087, 0.1396, 0.1650,
    0.0187, 0.7902, 0.6102, 0.1164, 0.8632, 0.4386, 0.6202, 0.4276,
    0.9163, 0.9785, 0.5454, 0.1289, 0.2046, 0.8605, 0.4218, 0.1642,
    0.2673, 0.8841, 0.6548, 0.6135, 0.4191, 0.4862, 0.1024, 0.1962,
    0.5313, 0.5920, 0.6284, 0.8210, 0.2920, 0.9228, 0.2764, 0.1430,
    0.7401, 0.7359, 0.8191, 0.6564, 0.5879, 0.2837, 0.7856, 0.3117
]

data_1 = data[:64]
data_2 = data[64:128]
data_3 = data[128:]

matrix_1 = torch.tensor(data_1).reshape((8, 8))
matrix_2 = torch.tensor(data_2).reshape((8, 8))
matrix_3 = torch.tensor(data_3).reshape((8, 8))

# 将三个矩阵组合成一个3x8x8的tensor
tensor = torch.stack((matrix_1, matrix_2, matrix_3)).reshape(1, 3, 8, 8).type(torch.float32)

# 打印tensor以验证结果
print(tensor.shape)




# Test MyLeNet5
model = CNN_No_MaxP()
model.load_state_dict(torch.load('/Users/den184/Documents/UNSW/SVF/test/SVF-Kane/NNs/SmallNet/CNN/No_MaxPool/save_model_CNN_No_MaxP/best_model_CNN_No_MaxP.pth'))
model.eval()  # 设置为评估模式

outputs_cnn1 = model(tensor)

output_dict = {}
for layer, output_tensor in outputs_cnn1.items():
    print(f"Output of {layer}: {output_tensor.shape}: {output_tensor}")
    output_dict[layer] = {"dim": str(output_tensor.shape), "values": output_tensor.tolist()}

with open('output.json', 'w') as f:
    json.dump(output_dict, f)