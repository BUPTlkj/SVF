import torch
import torch.nn as nn
from CNN_No_MaxP import CNN_No_MaxP

from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os


# 定义数据集大小
dataset_size = 1000

# 创建输入数据集，大小为 (dataset_size, 3, 8, 8)
input_dataset = torch.randn(dataset_size, 3, 8, 8)

num_classes = 4

labels = torch.randint(num_classes, (dataset_size,))




device = "cuda" if torch.cuda.is_available() else 'cpu'

model = CNN_No_MaxP().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(input_dataset, labels, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    
    for inputs, label in zip(input_dataset, labels):
        inputs = inputs.unsqueeze(0) 
        label = label.unsqueeze(0) 
        output = model(inputs)
        cur_loss = loss_fn(output, label)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(label == pred) / 1  # Divide by 1 since it's a single sample

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1

    print(f"train_loss: {loss / n:.4f}")
    print(f"train_acc: {current / n:.4f}")

def val(input_dataset, labels, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    
    with torch.no_grad():
        for inputs, label in zip(input_dataset, labels):
            inputs = inputs.unsqueeze(0)  
            label = label.unsqueeze(0)  
            output = model(inputs)
            cur_loss = loss_fn(output, label)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(label == pred) / 1

            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    print(f"val_loss: {loss / n:.4f}")
    print(f"val_acc: {current / n:.4f}")
    return current / n

epoch = 15
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n------------------')
    
    train(input_dataset, labels, model, loss_fn, optimizer)
    a = val(input_dataset, labels, model, loss_fn) 

    if a > min_acc:
        folder = 'save_model_CNN_No_MaxP'
        if not os.path.exists(folder):
            os.mkdir('save_model_CNN_No_MaxP')

        min_acc = a
        print('save best model original')

        torch.save(model.state_dict(), 'save_model_CNN_No_MaxP/best_model_CNN_No_MaxP.pth')

        # Specify a dummy input to the model
        dummy_input = torch.randn(1, 3, 8, 8, device='cpu')

        # Save the model as an ONNX file
        torch.onnx.export(model, dummy_input, 'save_model_CNN_No_MaxP/best_model_CNN_No_MaxP.onnx', input_names=['input'], output_names=['output'])

print('Done!')