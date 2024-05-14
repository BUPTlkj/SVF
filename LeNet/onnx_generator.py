import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def generate_data(num_samples=200, num_features=6, num_classes=4, filename='generated_data.csv'):
    # 生成数据
    data = []
    for _ in range(num_samples):
        label = np.random.randint(num_classes)  # 随机选择标签类别
        features = np.random.rand(num_features)  # 生成随机特征值
        row = [label] + features.tolist()
        data.append(row)

    # 转换为DataFrame
    columns = ['Label'] + [f'Feature_{i+1}' for i in range(num_features)]
    df = pd.DataFrame(data, columns=columns)

    # 保存到CSV文件
    df.to_csv(filename, index=False)

    print(f"数据已保存到 {filename} 文件中。")


# 定义第一个全连接神经网络模型
class FNNModel1(nn.Module):
    def __init__(self):
        super(FNNModel1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10) 

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x   
    
# class FNNModel1(nn.Module):
#     def __init__(self):
#         super(FNNModel1, self).__init__()
#         self.fc1 = nn.Linear(6, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 4)  # 假设有4个类别

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# 定义第二个全连接神经网络模型
class FNNModel2(nn.Module):
    def __init__(self):
        super(FNNModel2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# class FNNModel2(nn.Module):
#     def __init__(self):
#         super(FNNModel2, self).__init__()
#         self.fc1 = nn.Linear(6, 256)  # 输入层从28*28改为6，适应6个特征的数据
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 4)  # 输出层从10改为4，假设有4个类别

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# 定义第一个卷积神经网络模型
class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(14*14*32, 128)
        self.fc1_norm = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_norm(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class CNNModel1(nn.Module):
#     def __init__(self):
#         super(CNNModel1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 3), padding=0)  # 使用所有特征的卷积核
#         # 由于卷积核覆盖所有输入，输出的尺寸将是1x1x32，无需池化层
#         self.fc1 = nn.Linear(32, 128)  # 输入调整为单个卷积输出的展平形式
#         self.fc1_norm = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 4)  # 假设有4个类别

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = x.flatten(start_dim=1)  # 展平卷积层输出
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc1_norm(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# 定义第二个卷积神经网络模型
class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(28*28*32, 128)
        self.fc1_norm = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_norm(x))  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class CNNModel2(nn.Module):
#     def __init__(self):
#         super(CNNModel2, self).__init__()
#         # 设定卷积层，考虑到输入大小为1x2x3，这里使用小的卷积核
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 3), padding=0)  # 覆盖整个输入区域
#         # 由于卷积核大小和输入完全匹配，不需要池化层，卷积输出为1x1x32
#         self.fc1 = nn.Linear(32, 128)  # 调整为卷积输出展平后的大小
#         self.fc1_norm = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 4)  # 假设有4个类别

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = x.flatten(start_dim=1)  # 展平卷积层输出
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc1_norm(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# 训练模型的函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()  # 将模型设置为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


# 获取中间输出的函数
def forward_with_intermediate_outputs(model, x):
    outputs = {}
    x = model.norm1(x.view(x.size(0), -1))
    outputs['norm1'] = x.clone().detach()
    x = F.relu(model.fc1(x))
    outputs['fc1'] = x.clone().detach()
    x = model.norm2(x)
    outputs['norm2'] = x.clone().detach()
    x = F.relu(model.fc2(x))
    outputs['fc2'] = x.clone().detach()
    x = F.relu(model.fc3(x))
    outputs['fc3'] = x.clone().detach()
    x = model.fc4(x)
    outputs['fc4'] = x.clone().detach()
    return outputs, x


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.features = self.data_frame.iloc[:, 1:].values.astype(np.float32)  # 除去标签列的所有特征
        self.labels = self.data_frame.iloc[:, 0].values.astype(np.int64)  # 标签列

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return features, label


class CNNDataset(Dataset):

    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.features = self.data_frame.iloc[:, 1:].values.astype(np.float32)
        self.labels = self.data_frame.iloc[:, 0].values.astype(np.int64)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        # 重塑特征为1x2x3形状
        features = features.reshape(1, 2, 3)
        return features, label


# 主函数
if __name__ == '__main__':
    
    # 模拟数据
    # generate_data()

    # 设定变换以归一化数据
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换图片为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化处理
    ])

    # 下载并加载训练数据 Mnist
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # 使用CSVDataset加载数据
    # csv_data = CSVDataset('generated_data.csv')
    # train_loader = DataLoader(csv_data, batch_size=64, shuffle=True)

    # cnn_data = CNNDataset('generated_data.csv')
    # train_loader_cnn = DataLoader(cnn_data, batch_size=64, shuffle=True)

    # 实例化模型
    model1 = FNNModel1()
    model2 = FNNModel2()
    model3 = CNNModel1()
    model4 = CNNModel2()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
    optimizer4 = torch.optim.Adam(model4.parameters(), lr=0.001)

    # 训练模型
    train_model(model1, train_loader, criterion, optimizer1, num_epochs=5)
    train_model(model2, train_loader, criterion, optimizer2, num_epochs=5)
    train_model(model3, train_loader, criterion, optimizer3, num_epochs=5)
    train_model(model4, train_loader, criterion, optimizer4, num_epochs=5)

    # 为导出准备一个示例输入
    # input_ins = torch.randn(1, 1, 28, 28)  # 符合MNIST输入维度
    input_fnn = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])  # 手动指定6个特征的值
    input_cnn = torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]])  # 手动指定1x2x3形状的输入




    # 导出模型
    torch.onnx.export(model1, input_fnn, "model_fnn_1.onnx", export_params=True)
    torch.onnx.export(model2, input_fnn, "model_fnn_2.onnx", export_params=True)
    torch.onnx.export(model3, input_cnn, "model_cnn_1.onnx", export_params=True)
    torch.onnx.export(model4, input_cnn, "model_cnn_2.onnx", export_params=True)

    # 保存模型的状态字典为.pth文件
    torch.save(model1.state_dict(), 'model_fnn_1.pth')
    torch.save(model2.state_dict(), 'model_fnn_2.pth')
    torch.save(model3.state_dict(), 'model_cnn_1.pth')
    torch.save(model4.state_dict(), 'model_cnn_2.pth')