'''pytorch的dataset使用'''

from torch.utils.data import Dataset
from PIL import Image
import os

# class MyData(Dataset):
#
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir, self.label_dir)
#         self.img_path = os.listdir(self.path)
#
#     def __getitem__(self, idx):
#         img_name = self.img_path[idx]
#         img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
#         img = Image.open(img_item_path)
#         label = self.label_dir
#         return img, label
#
#     def __len__(self):
#         return len(self.img_path)
#
# root_dir = "train"
# ants_label_dir = "ants"
# ants_dataset = MyData(root_dir, ants_label_dir)
#
# bees_label_dir = "bees"
# bees_dataset = MyData(root_dir, bees_label_dir)
#
# train_dataset = bees_dataset + ants_dataset
#
# print(len(train_dataset), len(ants_dataset), len(bees_dataset))




class mydata(Dataset):

    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.path = os.path.join(self.root_dir, self.img_dir)
        self.lpath = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.label_path = os.listdir(self.lpath)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.img_path, self.label_path, img_name)
        img = Image.open(img_item_path)
        label = self.label_path[idx]
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "train"
ants_img_dir = "ants_image"
ants_label_dir = "ants_label"
ants_dataset = mydata(root_dir, ants_img_dir, ants_label_dir)