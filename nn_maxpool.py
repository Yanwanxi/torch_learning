import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)
        #False代表不足kernel的部分舍去，True代表不足也可以计算最大值

    def forward(self, input):
        output = self.maxpool(input)
        return output

cnn = CNN()
# output = cnn(input)
# print(output)

writer = SummaryWriter("nn_txt")

step = 0
for data in dataloader:
    imgs, targets = data
    output = cnn(imgs)
    writer.add_images("maxpool_output", output, step)
    step += 1

writer.close()