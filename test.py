import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./trainingdata/train/ants_image/0013035.jpg"
img = Image.open(image_path)
image = Image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

image = transform(image)

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

# 加载模型,如果现有机子只能cpu
model = torch.load("cnn_0.pth", map_location=torch.device('cpu'))
torch.reshape(image, (1, 3, 32, 32))
# 模型转化为测试类型
model.eval()
with torch.no_grad():
    output = model(image)

print(output.argmax(1))