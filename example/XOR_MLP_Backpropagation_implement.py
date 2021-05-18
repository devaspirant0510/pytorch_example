import torch

# 시그모이드
class Sigmoid:
    # 순전파 1/1+e^-x
    def forward(self,x:torch.Tensor)->torch.FloatTensor:
        self.out = 1/(1+torch.exp(-x))
        return 1/(1+torch.exp(-x))
    # 역전파 s*(1-s)
    def backward(self,grad=1) -> torch.FloatTensor:
        if self.out is None:
            return None
        return self.out*(1-self.out) * grad
# 손실함수 이진크로스엔트로피
class BinaryCrossEntropy:
    def forward(self,y,ypred):
        # log 0 방지하기 위해 delta 값 더함
        delta = 1e-7
        return -torch.sum((1-y)*(torch.log(1-ypred+delta)) + y * (torch.log(ypred+delta)))
    def backward(self,y,ypred):
        return ypred-y
# Layer 층에대한 w와 b 활성화 함수를 입력받음
class Layer:
    def __init__(self,w,b,activation=None):
        self.w = w
        self.b = b
        self.activation = activation
    # 활성화함수를 파라미터로 넣는지에따라 matmul 후 활성화 씌움
    def forward(self,x):
        self.x = x
        self.z = torch.matmul(x,self.w)+self.b
        if self.activation is None:
            return self.z
        else:
            self.s = self.activation.forward(self.z)
            return self.s
    # Todo: 추후 구현예정
    def backward(self,grad=1):
        delta = grad * self.activation.backward(self.z)
        self.dw = torch.matmul(delta.T,delta)
        self.db = torch.sum(delta,dim=0)
        self.dx = torch.matmul(delta,self.w.T)

# 학습 모델 클래스
class MyModel:
    # lr epoch 을 파라미터로 받음
    # batch size를 이용한 mini batch GD 는 추후 구현 예정
    def __init__(self,lr=0.1,iteration=10000,batch_size=None):
        self.layers = []
        self.loss = None
        self.lr = lr
        self.iteration = iteration
    # 모델에 레이어를 받아서 리스트에 저장함
    def add(self,layer):
        self.layers.append(layer)
    # 손실함수와 y 값을 저장함
    def set_loss(self,loss_func,y):
        self.loss = loss_func
        self.y = y
    # 손실함수를 출력해줌
    def get_loss(self):
        # forward porpgation 으로 out 값 구하고 loss 함수를 지정해야 가져올수 있음
        if self.out is None or self.loss is None:
            return None
        return self.loss.forward(self.y,self.out)
    # 레이어에 있는 값들을 가지고 propagation
    def forward(self,x):
        self.input = x
        for idx,layer in enumerate(self.layers):
            x = layer.forward(x)
        self.out = x
        return self.out
    # 오차역전파 구현
    # TODO : 추후 최적화 시킬 예정 Layer Class 의 backprop 활용
    def backwrad(self):
        #layer 2
        dloss = self.loss.backward(self.y,self.layers[1].s)
        dsig = self.layers[1].activation.backward() 
        loss2 = dloss * dsig
        #layer 2 dw
        self.dw2 = torch.matmul(self.layers[1].x.T,loss2)
        #layer 2 db
        self.db2 = torch.sum(loss2,dim=0)
        #layer 2 dx
        self.dx2 = torch.matmul(dloss,self.layers[1].w.T)

        #layer 1 dw
        loss1 = self.dx2 * self.layers[0].activation.backward()
        self.dw1 = torch.matmul(self.layers[0].x.T,loss1)
        self.db1 = torch.sum(loss1,dim=0)
    # train epoch 가지고 prop 과 backprop으로 학습시킴
    def train(self):
        for i in range(1,self.iteration+1):
            self.forward(self.input)
            self.backwrad()
            self.step()
            if i%100==0:
                print(f"epoch :{i}/{self.iteration}\t\t loss:{self.get_loss()}")
    # lr 만큼 업데이트 시킴
    def step(self):
        self.layers[1].w -= self.lr * self.dw2
        self.layers[1].b -= self.lr * self.db2
        self.layers[0].w -= self.lr * self.dw1
        self.layers[0].w -= self.lr * self.db1
    

x_train = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
y_train = torch.FloatTensor([[0],[1],[1],[0]])

layer1 = Layer(torch.rand(2,8),torch.rand(8),Sigmoid())
layer2 = Layer(torch.rand(8,1),torch.rand(1),Sigmoid())

model =  MyModel()
model.add(layer1)
model.add(layer2)
model.set_loss(BinaryCrossEntropy(),y_train)
model.forward(x_train)
model.train()
