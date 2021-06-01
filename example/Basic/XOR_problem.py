import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
y_train = torch.FloatTensor([[0],[1],[1],[0]])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        self.model_class = nn.Sequential(self.linear,self.sigmoid)
    def forward(self,x):
        x = self.model_class(x)
        return x
        # return self.sigmoid(self.linear(x))
                    
model = Model()

optimizer = optim.SGD(model.parameters(),lr=1)
cost_fun = nn.BCELoss()
epoch = 100

for i in range(1,epoch+1):
    hx = model(x_train)
    cost = F.binary_cross_entropy(hx,y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print(cost)
