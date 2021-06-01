import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# convolution 연산
# input size : 입력 필터의 크기 (iW,iH)
# filter size : 컨볼루션 필터의 크기(fW,fH)
# padding : 패딩의 두께 (p or (pW,pH))
# stride : 스트라이드 (스텝) (s or (sW,sH))
# output size : input size 로 convolution 연산을 했을때 나오는 shape

#               input size-filter + (2*padding)
#               ------------------------------- + 1
#                          stride

# width 와 height를 다른 값을 줄경우 output size의 width 와 height도 따로 구해야함

# torch.nn.Conv2d(in_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1)

#ex1
# input size 227,227
# filter size 11,11
# stride 4
# padding 0
# output size 227-11+(2*0)/4 +1 = 55

input_channel = torch.rand((1,1,227,227))*255
conv1 = nn.Conv2d(1,1,kernel_size=11,stride=4,padding=0)

output_channel = conv1(input_channel)
filter_ = list(conv1.parameters())[0].detach()


# pooling 연산
# -avg pooling : filter size 내에서 평균값을 넘김
# -max pooling : filter size 내에서 최댓값을 넘김 
# kernel size : 크기 만큼 pooling 연산
# stride : pooling 연산 할때 얼마나 겹칠지, 기본적으로 non-overlapping 됨 (겹치지 않음)
# padding : convolution 연산에서 padding 과 동일

# torch.nn.maxPool2d(kernal_size,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=Flase)

max_pooling = nn.MaxPool2d(2)
output = max_pooling(output_channel)
print(output.shape)

fig,ax = plt.subplots(1,4,figsize=(10,2))

ax[0].imshow(input_channel.view(227,227),cmap="gray")
ax[0].set_title("input")
ax[1].imshow(filter_.view(11,11),cmap="gray")
ax[1].set_title("filter")
ax[2].imshow(output_channel.detach().view(55,55),cmap="gray")
ax[2].set_title("conv1")
ax[3].imshow(output.detach().view(27,27),cmap="gray")
ax[3].set_title("max pooling")
plt.show()

