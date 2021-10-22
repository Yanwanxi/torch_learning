import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *


# 准备训练集
train_data = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
# 准备测试集
test_data = torchvision.datasets.CIFAR10("./trainingdata/CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
cnn = CNN()

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
'''1e-2'''
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练/测试次数
total_train_step = 0
total_test_step = 0

epoch = 10

# 添加tensorboard
writer = SummaryWriter("./torch-training/cnn_train")


for i in range(epoch):
    print("------------------第{}轮训练开始----------------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = cnn(imgs)
        loss = loss_function(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = cnn(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss
    print("整体测试集上的损失为:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(cnn, "cnn_{}.pth".format(i))
    print("模型已保存")


writer.close()