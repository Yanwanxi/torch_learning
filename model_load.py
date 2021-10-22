import torch
from model_save import *

# 方式1 --> 保存方式1
import torchvision

model1 = torch.load("vgg16_model1.pth")
print(model1)

# 方式2 --> 保存方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_model2.pth"))
# model2 = torch.load("vgg16_model2") '''读取的是字典形式的参数'''
print(vgg16)

# 方式1读取陷阱

model3 = torch.load("cnn_model1.pth")
print(model3)