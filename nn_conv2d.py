import torch
from torch import nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

cnn = CNN()


writer = SummaryWriter("nn_txt")
step = 0
for data in dataloader:
    imgs, targets = data
    output = cnn(imgs)
    print(output.shape)
    print(imgs.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    #torch.Size([64, 6, 30, 30])  6个通道无法显示

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1

writer.close()