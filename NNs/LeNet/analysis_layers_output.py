import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class FNNModel1(nn.Module):
    def __init__(self):
        super(FNNModel1, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        print('\n','\n')
        outputs = {}
        x = F.relu(self.fc1(x))
        outputs['fc1'] = x
        x = F.relu(self.fc2(x))
        outputs['fc2'] = x
        x = F.relu(self.fc3(x))
        outputs['fc3'] = x
        x = self.fc4(x)
        outputs['fc4'] = x
        return outputs

class FNNModel2(nn.Module):
    def __init__(self):
        super(FNNModel2, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        print('\n','\n')
        outputs = {}
        x = F.relu(self.fc1(x))
        outputs['fc1'] = x
        x = F.relu(self.fc2(x))
        outputs['fc2'] = x
        x = F.relu(self.fc3(x))
        outputs['fc3'] = x
        x = self.fc4(x)
        outputs['fc4'] = x
        return outputs

class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 3), padding=0)
        self.fc1 = nn.Linear(32, 128)
        self.fc1_norm = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        print('\n','\n')
        outputs = {}
        x = F.relu(self.conv1(x))
        outputs['conv1'] = x
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        outputs['fc1'] = x
        x = F.relu(self.fc1_norm(x))
        outputs['fc1_norm'] = x
        x = F.relu(self.fc2(x))
        outputs['fc2'] = x
        x = self.fc3(x)
        outputs['fc3'] = x
        return outputs

class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 3), padding=0)  # 覆盖整个输入区域
        self.fc1 = nn.Linear(32, 128)  # 卷积输出为1x1x32，调整为卷积输出展平后的大小
        self.fc1_norm = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 假设有4个类别

    def forward(self, x):
        print('\n','\n')
        outputs = {}
        x = F.relu(self.conv1(x))
        outputs['conv1'] = x
        x = x.flatten(start_dim=1)  # 展平卷积层输出
        x = F.relu(self.fc1(x))
        outputs['fc1'] = x
        x = F.relu(self.fc1_norm(x))
        outputs['fc1_norm'] = x
        x = F.relu(self.fc2(x))
        outputs['fc2'] = x
        x = self.fc3(x)
        outputs['fc3'] = x
        return outputs
    

class MyLeNet5(nn.Module):

    def __init__(self):
        super(MyLeNet5, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.c5 = nn.Conv2d(in_channels=16, out_channels=96, kernel_size=5)

        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(96, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        print('\n','\n')
        outputs = {}
        x = self.c1(x)
        outputs['conv1'] = x
        x = self.relu(x)
        outputs['relu'] = x
        x = self.s2(x)
        outputs['maxp1'] = x
        x = self.c3(x)
        outputs['conv2'] = x
        x = self.relu(x)
        outputs['relu2'] = x
        x = self.s4(x)
        outputs['maxp2'] = x
        x = self.c5(x)
        outputs['conv3'] = x
        x = self.flatten(x)
        outputs['flatten'] = x
        x = self.f6(x)
        outputs['fc1'] = x
        x = self.output(x)
        outputs['output'] = x
        return outputs


if __name__ == '__main__':

    # input_data_fnn = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # 示例输入数据

    # # Test FNN1 
    # model1 = FNNModel1()
    # model1.load_state_dict(torch.load('model_fnn_1.pth'))
    # model1.eval()  # 设置为评估模式，关闭Dropout等

    # outputs_fnn1 = model1(input_data_fnn)

    # for layer, output in outputs_fnn1.items():
    #     print(f"Output of {layer}:{output.shape}: {output}")

    # # Test FNN2
    # model2 = FNNModel2()
    # model2.load_state_dict(torch.load('model_fnn_2.pth'))
    # model2.eval()  # 设置为评估模式

    # outputs_fnn2 = model2(input_data_fnn)

    # for layer, output in outputs_fnn2.items():
    #     print(f"Output of {layer}:{output.shape} :{output}")



    # input_data_cnn = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]])  # 手动指定1x2x3形状的输入

    # # Test CNN1
    # model3 = CNNModel1()
    # model3.load_state_dict(torch.load('model_cnn_1.pth'))
    # model3.eval()  # 设置为评估模式，适用于推理和测试，禁用dropout等
    
    # outputs_cnn1 = model3(input_data_cnn)

    # for layer, output in outputs_cnn1.items():
    #     print(f"Output of {layer}: {output.shape}: {output}")

    # # Test CNN2
    # model4 = CNNModel2()
    # model4.load_state_dict(torch.load('model_cnn_2.pth'))
    # model4.eval()  # 设置为评估模式

    # outputs_cnn1 = model4(input_data_cnn)


    data = np.loadtxt('mnist_test.csv', delimiter=',')

    # Convert the first line into a tensor
    tensor = torch.from_numpy(data[0, 1:].reshape(1, 1, 28, 28).astype(np.float32))



    # Test MyLeNet5
    model5 = MyLeNet5()
    model5.load_state_dict(torch.load('/Users/den184/Documents/UNSW/SVF/SVF-Kane/LeNet/save_model_original_MaxPool2d/best_model_MaxPool2d.pth'))
    model5.eval()  # 设置为评估模式

    outputs_cnn1 = model5(tensor)

    output_dict = {}
    for layer, output_tensor in outputs_cnn1.items():
        print(f"Output of {layer}: {output_tensor.shape}: {output_tensor}")
        output_dict[layer] = {"dim": str(output_tensor.shape), "values": output_tensor.tolist()}

    with open('output.json', 'w') as f:
        json.dump(output_dict, f)
