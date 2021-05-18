import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


x_train = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
y_train = torch.FloatTensor([[0],[1],[1],[0]])

layer1 = nn.Linear(2,10)
layer2 = nn.Linear(10,1)
sigmod = nn.Sigmoid()
model = nn.Sequential(layer1,
                      sigmod,
                      layer2,
                      sigmod)
optimizer = optim.SGD(model.parameters(),lr=1)
cost_fun = nn.BCELoss()


epoch = 10000
for i in range(1,1+epoch):
    hx = model(x_train)
    cost = cost_fun(hx,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if i%100==0:
        print(f"epoch :{i}/{epoch}\t\t loss:{cost}")
