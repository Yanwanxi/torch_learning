import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 既保存了模型结构，又保存了模型参数
torch.save(vgg16, "vgg16_model1.pth")

# 保存方式2 保存模型参数，参数成字典状态（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_model2.pth")


#方式1的陷阱 --> 需要把保存的自己写的类定义到读取的py文件中
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        result = self.conv1(x)
        return result

cnn = CNN()

torch.save(cnn, "cnn_model1.pth")