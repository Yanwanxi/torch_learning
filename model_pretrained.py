import torchvision



# train_data = torchvision.datasets.ImageNet("./trainingdata/ImageNet", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_True = torchvision.models.vgg16(pretrained=True)
vgg16_False = torchvision.models.vgg16(pretrained=False)

# print(vgg16_True)

train_data = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
vgg16_True.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))

# print(vgg16_True)

# print(vgg16_False)
vgg16_False.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_False)