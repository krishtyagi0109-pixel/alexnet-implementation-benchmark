import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets,transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

from torchvision.datasets import ImageNet

from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = transform
)

train_loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle=True
)

from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
class AlexNet(nn.Module):

  def __init__(self):
    super().__init__()



    self.features = nn.Sequential(
        nn.Conv2d(3,96,kernel_size=11,stride = 4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride=2),

        nn.Conv2d(96,256,kernel_size=5,padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride=2),

        nn.Conv2d(256,384,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384,384,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384,256,kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,stride =2)



    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256*6*6,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,10)
    )

  def forward(self,x):
    x = self.features(x)
    x = self.fc(x)
    return x


image,label = next(iter(train_loader))

model = AlexNet().to(device)

output = model(image.to(device))

criterion = nn.CrossEntropyLoss()

loss = criterion(output,label.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=  0.0001)

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

print("Loss:", loss.item())
print("Gradient:", model.features[0].weight.grad.abs().mean().item())

epochs = 10

for epoch in range(epochs):
  model.train()
  tot_loss = 0
  for images,labels in train_loader:
    images,labels = images.to(device),labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    tot_loss += loss.item()

    avg_loss = tot_loss/len(train_loader)

  print(f"Epoch {epoch+1}: {avg_loss:.4f}")



#COMPARISON

from torchvision import models

tv_model = models.alexnet(weights = None)
tv_model.classifier[6] = nn.Linear(4096,10)
tv_model = tv_model.to(device)


print(f"my alexnet  params:- {sum(p.numel() for p in model.parameters())}")

print(f"Torch alexnet  params:- {sum(p.numel() for p in tv_model.parameters())}")

tv_criterion = nn.CrossEntropyLoss()
tv_optimizer = torch.optim.Adam(tv_model.parameters(), lr=0.0001)

tv_losses = []

for epoch in range(epochs):
    tv_model.train()
    tot_loss = 0
    for imgs, lbls in train_loader:

        imgs, lbls = imgs.to(device), lbls.to(device)
        .

        tv_optimizer.zero_grad()

        loss = tv_criterion(tv_model(imgs), lbls)

        loss.backward()
        tv_optimizer.step()

        tot_loss += loss.item()

    tv_losses.append(tot_loss / len(train_loader))

    print(f"Epoch {epoch+1}: {tv_losses[-1]:.4f}")

test_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor()

])

test_set = CIFAR10(root = "./data", train = False, download=True, transform=test_transform)

test_loader = DataLoader(test_set, batch_size = 128, shuffle=True)

classes = test_set.classes

#My Model

model.eval()
correct,total = 0,0

with torch.no_grad():
  for images , labels in test_loader:
    images,labels = images.to(device), labels.to(device)
    preds = model(images).argmax(dim = 1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)


#Torch model

tv_model.eval()
tv_correct, tv_total = 0, 0

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        p = tv_model(imgs).argmax(dim=1)
        tv_correct += (p == lbls).sum().item()
        tv_total += lbls.size(0)


print(f"accuracy:-{(correct/total)*100}")

tv_acc = 100 * tv_correct / tv_total
print(f"accuracy:-{tv_acc}")

