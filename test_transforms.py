from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "./trainingdata/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("transforms_txt")

'''transforms该如何被使用，将图片转化为tensor类型'''
tensor_trans = transforms.ToTensor() #选择一个class进行创建
tensor_img = tensor_trans(img)  #利用创建的工具进行转变

writer.add_image("ants", tensor_img)

writer.close()