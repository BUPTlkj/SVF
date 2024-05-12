import torch
from torch import nn
from LeNet import MyLeNet5

from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

data_transform = transforms.Compose([transforms.ToTensor()])


train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = MyLeNet5().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        
        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    print("train_loss" + str(loss/n))
    print("train_acc" + str(current/n))


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)

            cur_acc = torch.sum(y == pred)/output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    print("val_loss" + str(loss/n))
    print("val_acc" + str(current/n))

    return current/n

epoch = 15
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n------------------')
    
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn) 

    if a > min_acc:
        folder = 'save_model_original'
        if not os.path.exists(folder):
            os.mkdir('save_model_original')

        min_acc = a
        print('save best model original')

        # torch.save(model.state_dict(), 'save_model_original/best_model.pth')

        # Specify a dummy input to the model
        dummy_input = torch.randn(1, 1, 28, 28, device='cpu')

        # Save the model as an ONNX file
        torch.onnx.export(model, dummy_input, 'save_model_original/best_model.onnx', input_names=['input'], output_names=['output'])

print('Done!')