from torch import nn
import torch
import torch.nn.functional as F


'''要继承nn.Module的类'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


    def forward(self, input):
        output = input + 1
        return output

cnn = CNN()
x = torch.tensor(1.0)
output = cnn(x)
print(output)



'''卷积计算'''
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output1 = F.conv2d(input, kernel, stride=1)
print(output1)

output2 = F.conv2d(input, kernel, None, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=2, padding=1)
print(output3)