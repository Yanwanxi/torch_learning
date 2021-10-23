import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from model import *
import time



# 第二种使用GPU运行的方法

device = torch.device("cuda:0")

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
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output

cnn = CNN()
if torch.cuda.is_available():
    cnn = CNN.cuda()
# 第二种gpu运行
cnn = cnn.to(device)


# 损失函数
loss_function = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda()
# 第二种gpu运行
loss_function = loss_function.to(device)

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

start_time = time.time()
for i in range(epoch):
    print("------------------第{}轮训练开始----------------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 第二种gpu运行
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = cnn(imgs)
        loss = loss_function(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            # 第二种gpu运行
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cnn(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的损失为:{}".format(total_test_loss))
    print("整体测试集的正确率为:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(cnn, "cnn_{}.pth".format(i))
    print("模型已保存")


writer.close()