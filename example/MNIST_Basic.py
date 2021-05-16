import torchvision.datasets as dsets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import inspect
import time 

file_name   = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_       = file_name.split('.')[0]

MODEL_PATH  = f"{os.getcwd()}\\torch_model\\{file_}.pt"
MODEL_DIR   = f"{os.getcwd()}\\torch_model\\"

mnist_train = dsets.MNIST(root=os.getcwd()+"data/MNIST",train=True,transform=transforms.ToTensor(),download=True)
mnist_test  = dsets.MNIST(root=os.getcwd()+"data/MNIST",train=False,transform=transforms.ToTensor(),download=True)

data_loader = DataLoader(dataset=mnist_train,batch_size=100,shuffle=False,drop_last=True)

def draw_mnist():
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(20,20))
    
    for i in range(3):
        for j in range(3):
            idx = i*3+j
            data = list(data_loader)[0][0]
            label = list(data_loader)[0][1]
            ax[i][j].imshow(data[idx].view(28,28),cmap="gray")
            ax[i][j].set_title(int(label[idx]))
            print(idx)
    plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_mnist_plot.png")
    plt.show()

def draw_acc_loss(acc,loss):
    plt.plot(loss)
    plt.plot(acc)
    plt.show()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28,10)
    def forward(self,x):
        return self.linear(x)
        

def accuarcy(x,y):
    x_pred = torch.argmax(x,dim=1)
    acc = torch.sum(x_pred==y)
    return acc/len(x)

def train_mnist():
    start_time = time.time()
    for i in range(1,epoch+1):
        for idx,[x,y] in enumerate(data_loader):
            x=x.view(-1,28*28)
            hx = model(x)
            cost = cost_fun(hx,y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            acc = accuarcy(F.softmax(hx,dim=1),y)

            cost_li.append(cost.detach())
            acc_li.append(acc)

            if idx%49==0:
                print(f"epoch : {i}/{epoch} \t\t cost : {cost} \t\t accuracy : {acc}%")

    print(model.state_dict())
    torch.save(model.state_dict(),f"{os.getcwd()}\\torch_model\\{file_}.pt")
    end_time = time.time()-start_time
    print("training : ",end_time,"s")
    draw_acc_loss(acc_li,cost_li)

def test_data(load_model):
    linear_weight = load_model['linear.weight']
    linear_bias   = load_model['linear.bias']
    model.load_state_dict(torch.load(MODEL_PATH))
    model_pre  = 0 
    model_size = 10000
    with torch.no_grad():
        for idx,[x,y] in enumerate(mnist_test):
            x = x.view(-1,28*28)
            hx = model(x)
            model_pre += torch.sum(torch.argmax(hx,dim=1)==y)
        print(model_pre/model_size)


epoch     = 10
lr        = 0.05
model     = Model()
optimizer = optim.SGD(model.parameters(),lr=lr)
cost_fun  = nn.CrossEntropyLoss()
cost_li   = []
acc_li    = []

if f"{file_}.pt" in os.listdir(MODEL_DIR):
    load_model    = torch.load(MODEL_PATH)
    test_data(load_model)
    
else:
    train_mnist()

    

