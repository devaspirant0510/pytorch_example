import torch
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import inspect
plt.style.use("seaborn-whitegrid")
file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
def draw_init_plot(x_train,y_train):
    file_ = file_name.split('.')[0]
    plt.scatter(x_train,y_train,marker="o",color="red",alpha=0.5)
    plt.xlabel("x train")
    plt.ylabel("y label")
    plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_init_plot.png")


x_train = torch.FloatTensor([1,2,3,4,5]).view(-1,1)
y_train = torch.FloatTensor([2.3,3.1,4.5,5.3,6.6]).view(-1,1)

draw_init_plot(x_train,y_train)
# require_gard는 weight 와 bias를 학습시킴
W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
hx = x_train.matmul(W)+b

loss = torch.mean((hx-y_train)**2)

optimizer = optim.SGD([W,b],lr=0.01)
fig,ax = plt.subplots(nrows=5,ncols=1,figsize=(50,10))
epoch = 1000
for i in range(1001):
    hx = x_train.matmul(W)+b
    loss = torch.mean((hx-y_train)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%200==0:
        idx = i//200
        # ax[idx].scatter(x_train,y_train,marker="o",color="red",alpha=0.5)
        # ax[idx].plot(x_train.numpy(),hx.numpy())
        print(loss)

loss = torch.mean((hx-y_train)**2)
W1 = W[0]
print(torch.cuda.is_available())