import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision import datasets
import os
import time
import inspect
import matplotlib.pyplot as plt
import seaborn as sns

file_name   = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_       = file_name.split('.')[0]

MNIST_train = datasets.MNIST(root=os.getcwd()+"data/MNIST",train=True,transform=transforms.ToTensor(),download=True)
MNIST_test = datasets.MNIST(root=os.getcwd()+"data/MNIST",train=False,transform=transforms.ToTensor(),download=True)

x_train = MNIST_train.data.view(60000,28*28)
y_train = MNIST_train.targets

batch_size = 100
epoch = 3
lr = 0.01

dataloader = DataLoader(MNIST_train,batch_size=batch_size,shuffle=True,drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28,300)
        self.linear2 = nn.Linear(300,100)
        self.linear3 = nn.Linear(100,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


def accuracy(y_pred,y_true):
    y_true = y_true.view(100)
    
    y_pred = F.softmax(y_pred,dim=1)
    
    acc = torch.argmax(y_pred,dim=1)
    score = torch.sum(y_true==acc)
    size = len(y_true)
    return score/size


model = Model()
cost_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

accuracy_list = []
loss_list = []

accuracy_score = 0
loss_score = 0

for ep in range(1,epoch+1):
    accuracy_score = 0
    loss_score = 0
    for x,y in dataloader:
        x = x.view(100,28*28)
        pred = model(x)
        loss = cost_func(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_score = accuracy(pred,y)
        loss_score+=loss

        accuracy_list.append(acc_score.detach())
        loss_list.append(loss.detach())


    loss_score/=batch_size
    
    print(f"epoch : {ep}/{epoch}\t\tloss:{loss_score}\t\taccuracy:{acc_score}")

plt.plot(accuracy_list)
plt.plot(loss_list)
plt.show()