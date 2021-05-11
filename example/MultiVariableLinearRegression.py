import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import inspect
import os

plt.style.use("seaborn-whitegrid")

file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_ = file_name.split('.')[0]

x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

def NoModule():

    optimizer = optim.SGD([W,b],lr=0.00001)

    epoch = 20

    for i in range(1,epoch+1):
        hx = x_train.matmul(W)+b
        cost = torch.mean((hx-y_train)**2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(f"epoch : {i}/{epoch} \t\t cost : {cost} \t\t hx:{hx.squeeze().detach()}")

W = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

class MultiLinearReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)

model = MultiLinearReg()
optimizer = optim.SGD(model.parameters(),lr=0.00001)

epoch = 30
plot_x = range(epoch)
plot_cost = []
for i in range(1,epoch+1):
    hx = model(x_train)
    cost = F.mse_loss(hx,y_train)
    plot_cost.append(cost.detach())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(f"epoch : {i}/{epoch} \t\t cost : {cost} \t\t hx:{hx.squeeze().detach()}")

plt.plot(plot_x,plot_cost,'--r')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_loss.png")