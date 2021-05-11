import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data =  [[73,80,75],
                        [93,88,93],
                        [89,91,90],
                        [96,98,100],
                        [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])
        return x,y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)

model = Model()
optimizer = optim.Adagrad(model.parameters(),lr=0.1)

dataset = CustomDataset()

dataloader = DataLoader(
    dataset,
    batch_size=5,
    shuffle=True)

print(list(dataloader))
epoch = 30
for i in range(1,epoch+1):
    for idx,batch in enumerate(dataloader):
        x_train,y_train = batch
        hx = model(x_train)
        
        loss = F.mse_loss(hx,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        


