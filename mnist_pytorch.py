import torch
import torch.nn as nn
from torchvision import datasets,transforms

#将数据集转换为tensor类型
transform = transforms.Compose([transforms.ToTensor()])
#加载数据并转换（必须要转换，否则无法直接使用图片数据）
dataset = datasets.MNIST(root='torch_data/',train=True,download=True,transform=transform)
#loader加载数据并分批
loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=100)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #与tensorflow基本类似，但是pytorch的层要指定输入和输出，tf只需要指定输出。。
        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(28*28,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,10)
            )
    def forward(self,input):
        return self.network(input)

model = Model()
print(model)
device = torch.device('cuda:0') #使用gpu
loss_fn = torch.nn.CrossEntropyLoss().to(device)    #损失函数（pytorch似乎没有sparse的分类损失函数）
optimizer = torch.optim.Adam(model.parameters())    #优化器随便选一个
for epoch in range(5):
    total_loss = 0.
    corrects = 0
    
    for x,y in loader:
        #这里与tensorflow的GradientTape类似，将需要求导的部分包起来（注意反向传播也要包在里面）
        optimizer.zero_grad()
        x = torch.flatten(x,start_dim=1,end_dim=3)  #同样先将图片数据打平
        preds = model(x)    #前向传播
        loss = loss_fn(preds,y) #计算损失
        loss.backward() #反向传播
        optimizer.step()

        #统计损失值和准确率
        total_loss += loss.item()
        preds = torch.argmax(preds,dim=1)
        corrects += torch.sum(preds==y)

    print('epoch',epoch,'loss',total_loss,'acc',(100.*corrects/len(dataset)).item())