# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import torch
from vgg import *
from PCA import *
import torch.nn.functional as F
import numpy as np
from LCC import largest_connect_component

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default="download/", help='path of images')
parser.add_argument('--save-dir', type=str, default="output/", help='output path')
parser.add_argument('--train', type=str, default="train_jiezhi/", help='for training')
parser.add_argument('--test', type=str, default="test_jiezhi/", help='for testing')

opt = parser.parse_args()
print(opt)

in_dir = opt.data_dir + opt.train
out_dir = opt.save_dir + opt.train

# 读取图片

image_files = os.listdir(in_dir) + os.listdir(opt.data_dir + opt.test)

images = []
for idx, image_file in enumerate(image_files):
    # 读入，统一分辨率，转为RGB，转为Tensor，把224,224,3变为3,224,224
    # 再变为1,3,224,224
    if idx < 50:
        image = cv2.imread(os.path.join(in_dir, image_file))
    else:
        image = cv2.imread(os.path.join(opt.data_dir + opt.test, image_file))
    # 主流的深度学习框架支持224,224的图片
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将RGB通道转到第一个维度，方便后续模型的读入。
    # 最终转化为 (1,3,224,224)
    image = torch.from_numpy(image)
    image = image.float()
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    images.append(image)

num_image = len(images)

# 将图片喂到预训练的CNN模型中得到结果
model_CNN = vgg19(pretrained=True)
# (num_image,512,14,14)
descriptors = []
for i in range(0, num_image):
    descriptor = model_CNN(images[i])[0]
    descriptors.append(descriptor)
    # descriptors = model_CNN(cat_images)[0]
descriptors = torch.cat(descriptors)

# 计算PCA
pca = PCA()
# (num_image,14,14)


# 划分训练集和测试集
descriptors_train = descriptors[:50]
descriptors_test = descriptors[50:]

# 训练PCA的参数
pca(descriptors_train)
projected_descriptors = pca.predict(descriptors_test)

# 这里开始做测试,

# 将所有小于0的值用0代替
projected_descriptors = torch.clamp(projected_descriptors, min=0)
# 归一化为[0,1]
# maxv : (n,1,1)
max_mat = projected_descriptors.view(num_image / 2, -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2)
normalized_project = projected_descriptors / max_mat

# project_map : 插值 (n,14,14) -> (n,1,width,height)
normalized_project = F.interpolate(normalized_project.unsqueeze(1), size=(224, 224), mode='bilinear',
                                   align_corners=False) * 255.
# 发现用nearest插值效果并不好
# normalized_project = F.interpolate(normalized_project.unsqueeze(1), size=(224, 224), mode='nearest') * 255.

# 输出
origins = []
masks = []
rectangle_images = []
colormap_masks = []
weighted_images = []

for i, name in enumerate(image_files[50:]):
    # 获取原图
    # image = cv2.imread(os.path.join(in_dir, name))
    image = cv2.imread(os.path.join(opt.data_dir + opt.test, name))
    # 主流的深度学习框架支持224,224的图片
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origins.append(image)

    image_copy = image.copy()

    # 掩码
    mask = normalized_project[i].repeat(3, 1, 1).permute(1, 2, 0)
    mask_copy = mask.clone()
    mask = mask.detach().numpy()
    masks.append(mask)

    # 赋值掩码并转为灰度图
    mask_copy = mask_copy.detach().numpy().astype(np.uint8)
    gray_image = cv2.cvtColor(mask_copy, cv2.COLOR_RGB2GRAY)

    # 将掩码转为二值图并找到最大连通区域并绘制
    ret, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    min_x, max_x, min_y, max_y = largest_connect_component(binary_image)
    rectangle_image = cv2.rectangle(image_copy, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    rectangle_images.append(rectangle_image)

    # 类似于热点图
    colormap_mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    colormap_masks.append(colormap_mask)

    # 原图与热点图混合
    weighted_image = cv2.addWeighted(image, 0.5, colormap_mask, 0.5, 0.0)
    weighted_images.append(weighted_image)

out_dir = out_dir.rstrip("\\")
isExists = os.path.exists(out_dir)
if not isExists:
    # 如果不存在则创建目录
    os.makedirs(out_dir)

origins = np.concatenate(origins, 1)
cv2.imwrite(out_dir + 'origin.jpg', origins)

masks = np.concatenate(masks, 1)
cv2.imwrite(out_dir + 'masks.jpg', masks)

rectangle_images = np.concatenate(rectangle_images, 1)
cv2.imwrite(out_dir + 'rectangle_images.jpg', rectangle_images)

colormap_masks = np.concatenate(colormap_masks, 1)
cv2.imwrite(out_dir + 'colormap_masks.jpg', colormap_masks)

weighted_images = np.concatenate(weighted_images, 1)
cv2.imwrite(out_dir + 'weighted_images.jpg', weighted_images)

all_together = np.concatenate((origins, masks, rectangle_images, colormap_masks, weighted_images), 0)
cv2.imwrite(out_dir + 'all.jpg', all_together)
