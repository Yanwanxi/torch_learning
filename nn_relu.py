import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(datasets, batch_size=64, shuffle= True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = ReLU(False)
        self.sigmoid = Sigmoid()
        #True表示原始数据被代替input=-1，input=0，False表示input=-1，output=0

    def forward(self, input):
        output = self.sigmoid(input)
        return output

cnn = CNN()

writer = SummaryWriter("nn_txt")
step = 0
for data in dataloader:
    imgs, targets = data
    output = cnn(imgs)
    writer.add_images("output_sigmoid", output, step)
    step += 1

writer.close()