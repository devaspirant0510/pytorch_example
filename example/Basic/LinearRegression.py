import torch
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import inspect
plt.style.use("seaborn-whitegrid")

file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_ = file_name.split('.')[0]

def draw_init_plot(x_train,y_train):
    plt.scatter(x_train,y_train,marker="o",color="red",alpha=0.5)
    plt.xlabel("x train")
    plt.ylabel("y label")
    plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_init_plot.png")

x_train = torch.FloatTensor([1,2,3,4,5]).view(-1,1)
y_train = torch.FloatTensor([2.3,2.9,4.1,5.4,5.7]).view(-1,1)
# draw_init_plot(x_train,y_train)
# require_grad는 weight 와 bias를 학습시킴
W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
hx = x_train.matmul(W)+b

loss = torch.mean((hx-y_train)**2)

optimizer = optim.Adagrad([W,b],lr=0.01)
fig,ax = plt.subplots(nrows=6,ncols=1,figsize=(4,25))
fig.subplots_adjust(hspace = 1)
epoch = 4000
for i in range(epoch):
    hx = x_train.matmul(W)+b
    loss = torch.mean((hx-y_train)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%800==0:
        idx = i//800
        hx_list = list(map(float,hx))
        ax[idx].scatter(x_train,y_train,marker="o",color="red",alpha=0.5)
        ax[idx].plot(x_train,hx_list)
        ax[idx].set_title("epcoh : %d" % (i))
        print(loss)

hx_list = list(map(float,hx))
ax[5].plot(x_train,hx_list)
ax[5].scatter(x_train,y_train,marker="o",color="red",alpha=0.5)
ax[5].set_title("output")
plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_training_plot.png")