
'''利用tensorboard进行可视化，观察每一轮迭代的结果'''
'''主要涉及到两个函数，SummaryWriter(文件名) 
   add_image("名称"，图像， 轮数， 图像格式)
   add_scalar("名称"， y， x)
'''

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os
print(os.getcwd())

writer = SummaryWriter("tensorboard_txt")
image_path = "./trainingdata/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')

# y = 2 * x

for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)



writer.close()