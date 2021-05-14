import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#===================# Softmax #===================#
"""
z = torch.FloatTensor([[1,2,3],[4,5,1]])
# 1차원 벡터일때는 dim=0
# 2차원 행렬일때는 dim=1
hx = F.softmax(z,dim=1)
torch.manual_seed(2)
print(hx)
print(torch.sum(hx,dim=1))

z =torch.rand((3,3))
ex = torch.zeros_like(z)
print(ex)
ex.scatter_(1,torch.LongTensor([[0],[1],[2]]),1)
print(ex)
"""
#===================# Cross Entropy #===================#
"""
z = torch.rand((3,3))
hx = F.softmax(z,dim=1)
cost = (ex*-torch.log(hx)).mean(dim=1).sum()
print(cost)
print(F.cross_entropy(F.log_softmax(z,dim=1),torch.LongTensor([0,1,2])))
print(F.nll_loss(torch.log(hx),torch.LongTensor([0,1,2])))

print(F.log_softmax(z,dim=1))
"""

#===================# Model Setting #===================#

x_data = [[1,2,1,1],
           [2,1,3,2],
           [3,1,3,4],
           [4,1,5,5],
           [1,7,5,5],
           [1,2,5,6],
           [1,6,6,6],
           [1,7,7,7]]
y_data = [2,2,2,1,1,1,0,0]

se1 = pd.DataFrame(x_data)
se2 = pd.DataFrame(y_data)
df = pd.concat([se1,se2],axis=1)
df.columns = ['x1','x2','x3','x4','y']

#===================# Data Visualization #===================#
# sns.pairplot(data=df,hue='y')
# plt.show()
#===================# Softmax Classification #===================#
x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)

#model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.linear(x))

model = Model()
#optimizer 
W = torch.zeros((4,3),requires_grad=True)
b = torch.zeros(3,requires_grad=True)
optimizer = optim.Adagrad(model.parameters(),lr=0.1)
epoch = 10000
for i in range(1,epoch+1):
    hx = model(x_train)
    cost = F.cross_entropy(hx,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if i%100==0:
        print(f"{i}/{epoch}\t\t cost:{cost}")
