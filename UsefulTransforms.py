from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("transforms_txt")
img_path = "./trainingdata/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)

#  ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("ants_tensor", img_tensor)

#Normalize

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([2, 2, 3], [7, 4, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize

trans_resize = transforms.Resize((512, 512))
# img PIL --> resize --> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL --> totensor --> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize, 0)
# print(img_resize)

#Compose - resize -2

trans_resize_2 = transforms.Resize(512)
# PIL --> PIL --> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)


#RandomCrop

trans_random = transforms.RandomCrop((16,64))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()
