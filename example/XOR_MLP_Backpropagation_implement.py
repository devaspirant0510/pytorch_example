import torch


class Sigmoid:
    def forward(self,x:torch.Tensor)->torch.FloatTensor:
        self.out = 1/(1+torch.exp(-x))
        return 1/(1+torch.exp(-x))

    def backward(self,grad=1) -> torch.FloatTensor:
        if self.out is None:
            return None
        return self.out*(1-self.out) * grad

class BinaryCrossEntropy:
    def forward(self,y,ypred):
        delta = 1e-7
        return -torch.sum((1-y)*(torch.log(1-ypred+delta)) + y * (torch.log(ypred+delta)))
    def backward(self,y,ypred):
        return ypred-y

class Layer:
    def __init__(self,w,b,activation=None):
        self.w = w
        self.b = b
        self.activation = activation
    def forward(self,x):
        self.x = x
        self.z = torch.matmul(x,self.w)+self.b
        if self.activation is None:
            return self.z
        else:
            self.s = self.activation.forward(self.z)
            return self.s

    def backward(self,grad=1):
        delta = grad * self.activation.backward(self.z)
        self.dw = torch.matmul(delta.T,delta)
        self.db = torch.sum(delta,dim=0)
        self.dx = torch.matmul(delta,self.w.T)



class MyModel:
    def __init__(self,lr=0.1,iteration=10000,batch_size=None):
        self.layers = []
        self.loss = None
        self.lr = lr
        self.iteration = iteration

    def add(self,layer):
        self.layers.append(layer)

    def set_loss(self,loss_func,y):
        self.loss = loss_func
        self.y = y

    def get_loss(self):
        if self.out is None or self.loss is None:
            return None
        return self.loss.forward(self.y,self.out)

    def forward(self,x):
        self.input = x
        self.layers[0].x = self.input
        self.W1 = self.layers[0].w
        self.b1 = self.layers[0].b
        self.z1 = torch.matmul(self.input,self.W1)+self.b1
        self.s1 = self.layers[0].activation.forward(self.z1)
        self.layers[0].s = self.s1
        self.layers[1].x = self.s1

        self.W2 = self.layers[1].w
        self.b2 = self.layers[1].b

        self.z2 = torch.matmul(self.s1,self.W2)+self.b2
        self.s2 = self.layers[1].activation.forward(self.z2)
        self.layers[1].s = self.s2
        # for idx,layer in enumerate(self.layers):
        #     if idx==0:
        #         x = layer.forward(self.input)
        #     else:
        #         x = layer.forward(x)
        self.out = self.s2
        return self.out

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

    def train(self):
        for i in range(1,self.iteration+1):
            self.forward(self.input)
            self.backwrad()
            self.step()
            if i%100==0:
                print(f"epoch :{i}/{self.iteration}\t\t loss:{self.get_loss()}")
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
