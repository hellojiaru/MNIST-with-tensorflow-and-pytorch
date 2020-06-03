import torch
import torch.nn as nn
from torchvision import datasets,transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='torch_data/',train=True,download=True,transform=transform)
loader = torch.utils.data.DataLoader(dataset,100)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            
            torch.nn.Linear(28*28,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,10)
            )
    def forward(self,input):
        return self.network(input)

model = Model()
print(model)
device = torch.device('cuda:0')
loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(5):
    total_loss = 0.
    corrects = 0
    
    for x,y in loader:
        optimizer.zero_grad()
        x = torch.flatten(x,start_dim=1,end_dim=3)
        preds = model(x)
        loss = loss_fn(preds,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(preds,dim=1)
        corrects += torch.sum(preds==y)

    print('loss',total_loss,'acc',(100.*corrects/len(dataset)).item())