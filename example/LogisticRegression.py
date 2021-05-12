import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import matplotlib.pyplot as plt
import seaborn as sns
import os
import inspect

plt.style.use("seaborn-whitegrid")

file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_ = file_name.split('.')[0]

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# hx = 1/(1+torch.exp(x_train.matmul(W)+b))
# m=len(x_train)

# cost = (1/m)*-torch.sum((1-y_train)*torch.log(1-hx)+y_train*torch.log(hx))
# costF = F.binary_cross_entropy(hx,y_train)


# optimizer = optim.Adagrad([W,b],lr=0.5)
# epoch = 2000
# loss_li = []
# for i in range(1,epoch+1):
#     hx = sigmoid(x_train.matmul(W)+b)
#     cost = F.binary_cross_entropy(hx,y_train)

#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#     loss_li.append(cost.detach())
#     if i%100==0:
#         print(f"epoch : {i}/{epoch}\t\t cost : {cost}")

# loss_x = range(epoch)

class Modle(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.linear(x))

def accuracy(hx,y_train):
    predict = hx>=torch.FloatTensor([0.5])
    predict = predict.float()
    print(predict)
    return torch.sum(predict==y_train)/len(hx)

def draw_chart(loss_x,loss_li):
    plt.plot(loss_x,loss_li,"--r",alpha=0.5,lw=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_loss_plot.png")

model = Modle()

optimizer = optim.SGD(model.parameters(),lr=1)
epoch = 200
for i in range(1,epoch+1):
    hx = model(x_train)
    cost = F.binary_cross_entropy(hx,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if i%20==0:
        print(f"epoch : {i}/{epoch}\t\tcost : {cost}\t\tAccuracy : {accuracy(hx,y_train)}")



