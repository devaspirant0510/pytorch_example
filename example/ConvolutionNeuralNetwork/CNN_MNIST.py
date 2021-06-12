import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os
import inspect
print(torch.cuda.is_available())
file_name = inspect.getfile(inspect.currentframe()).split('/')[-1]
file_ = file_name.split('.')[0]

MNIST_TRAIN = datasets.MNIST(root=os.getcwd() + "/data/MNIST", train=True, transform=transforms.ToTensor(),
                             download=True)
MNIST_TEST = datasets.MNIST(root=os.getcwd() + "/data/MNIST", train=False, transform=transforms.ToTensor(),
                            download=True)

MODEL_PATH = f"{os.getcwd()}\\torch_model\\{file_}.pt"
MODEL_DIR = f"{os.getcwd()}\\torch_model\\"

learning_rate = 1e-5
batch_size = 100
epoch = 10

dataloader = DataLoader(MNIST_TRAIN, batch_size=batch_size, shuffle=True, drop_last=True)

x_train, y_train, x_test, y_test = (MNIST_TRAIN.data.view(-1, 28 * 28),
                                    MNIST_TRAIN.targets,
                                    MNIST_TEST.data.view(-1, 28 * 28),
                                    MNIST_TEST.targets)


def check_MNIST_data(x_data, y_data, data_size: tuple = (2, 2), view: bool = True, download: bool = True) -> None:
    fig, ax = plt.subplots(nrows=data_size[0], ncols=data_size[1])
    for i in range(data_size[0]):
        for j in range(data_size[1]):
            idx = np.random.randint(0, len(x_data))
            draw_x = x_data[idx, :]
            draw_y = y_data[idx]
            ax[i, j].imshow(draw_x.view(28, 28), cmap="gray")
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)
            ax[i, j].set_title(int(draw_y))

    if view:
        plt.show()
    if download:
        plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_check_mnist_data.png")
        print(f"{os.getcwd()}\\source\\{file_}_check_mnist_data.png 로 저장 되었습니다.")


def batch_accuarcy(pred, ytrue):
    argmax = torch.argmax(pred, dim=1)
    correct = torch.sum(argmax == ytrue)
    return correct / len(pred)


# check_MNIST_data(x_train,y_train,download=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Linear(7 * 7 * 64, 10, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out


model = Model()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
cost_func = nn.CrossEntropyLoss()

cost_li = []
acc_li = []
for ep in range(1, epoch + 1):
    for x, y in dataloader:
        pred = model(x)
        acc_score = batch_accuarcy(pred, y)

        cost = cost_func(pred, y)

        acc_li.append(acc_score)
        cost_li.append(cost.detach())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print(f"epoch:{ep}/{epoch}\t\tcost{cost}\t\taccuarcy:{acc_score}%")

torch.save(model.state_dict(), f"{os.getcwd()}\\torch_model\\{file_}.pt")

plt.plot(acc_li)
plt.plot(cost_li)
plt.savefig(f"{os.getcwd()}\\example\\source\\{file_}_accuarcy_and_cost.png")
