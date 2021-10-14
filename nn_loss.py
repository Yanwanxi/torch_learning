import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, MSELoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# inputs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
# targets = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
#
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
#
# loss1 = L1Loss(reduction='sum')
# result = loss1(inputs, targets)
#
# loss_mse = MSELoss()
# result_MSE = loss_mse(inputs, targets)
#
# print(result)
# print(result_MSE)


dataset = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):

        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
cnn = CNN()
optim = torch.optim.SGD(cnn.parameters(), lr=0.01,)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = cnn(imgs)
        # print(output)
        # print(targets)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()  #反向传播
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)

