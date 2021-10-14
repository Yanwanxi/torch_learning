import torchvision

# 测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

#测试数据集中的第一张图片以及target
# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
    #   print(imgs.shape)
    #   print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, global_step=step)
        step += 1
#   print(imgs.shape)
writer.close()