import torchvision
from torch.utils.tensorboard import SummaryWriter

'''也可以用torchvision.transforms.Compose([xxx, xxx])来转化数据类型'''
dataset_transform = torchvision.transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root="./trainingdata/CIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./trainingdata/CIFAR10", train=False, transform=dataset_transform, download=True)

print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
#
# print(test_set.classes[target])

# img.show()

writer = SummaryWriter("dataset_transforms_txt")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()