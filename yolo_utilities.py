import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import h5py
import matplotlib.image as mpimg
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import json
# import xmltodict
import sys
from skimage import io
# import neccessary packages
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸

    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    # image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式

    return new_image


def squarePadImg(img, size):
    ow, oh = img.size
    if ow == oh:
        return img.resize(size)
    padw = False if ow > oh else True
    pad = (ow, ow) if not padw else (oh,oh)
    ig = img
    return pad_image(ig,pad).resize(size)


def drawPred (img, arry, thre = 0.5):
    size = img.size
    imgcp = img
    draw = ImageDraw.Draw(imgcp)
    arry = arry.squeeze()
    gridS = size[0] / 13
    for i in range(13):
        for j in range(13):
            if arry[i][j][2] > thre or arry[i][j][3] > thre:

                outline = (255, 0, 0) if arry[i][j][1] > arry[i][j][0] else (0, 255, 0)

                xc1 = i * gridS + arry[i][j][4] * gridS
                yc1 = j * gridS + arry[i][j][5] * gridS
                w1 = arry[i][j][6] * size[0]
                h1 = arry[i][j][7] * size[0]

                xc2 = i * gridS + arry[i][j][8] * gridS
                yc2 = j * gridS + arry[i][j][9] * gridS
                w2 = arry[i][j][10] * size[0]
                h2 = arry[i][j][11] * size[0]

                draw.rectangle( [xc1 - w1/2, yc1 - h1/2, xc1 + w1/2, yc1 + h1/2], outline = outline, width = 1)
                draw.rectangle( [xc2 - w2/2, yc2 - h2/2, xc2 + w2/2, yc2 + h2/2], outline = outline, width = 1)

    return imgcp



class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, picture_list, transform=None, img_size = 416, transformProb = 10):
        # 1. Initialize file path or list of file names.
        self.picture_list = picture_list
        self.transform = transform
        self.img_size = img_size
        self.transformProb = transformProb
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        image = Image.open(self.picture_list[index])
        label = self.getLabel(image.size, ".".join(self.picture_list[index].split(".")[0:-1]) + ".txt")
        image = self.squarePadImg(image, (self.img_size, self.img_size))
        
        r = random.randint(1,self.transformProb)
        if self.transform and r == 1:
            image = self.transform(image)

        image = transforms.ToTensor()(image)
        
        sample = {"image":image[0:3], "label":label}
        return sample
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.picture_list)

    def getLabel(self, size, path):
        ow, oh = size
        nw, nh = size
        d = abs(ow- oh)/2
        
        padw = False if ow > oh else True
        if padw:
            nw = nw + 2*d
        else:
            nh = nh + 2*d

        lbl = torch.zeros(13,13,12)

        for line in open(path):
            args = line.replace("\n", "").split(" ")

            if len(args) < 4:
                continue
            c, xc, yc, w, h = int(args[0]), float(args[1]) * ow, float(args[2]) * oh, float(args[3]) * ow, float(args[4]) * oh
            c1, c2 =0, 0

            if "Mask_" in path :
                c = 2 - c
            
            if c >= 1:
                c1 = 1
            else:
                c2 = 1
            if padw:
                xc = xc + d
            else:
                yc = yc + d

            w, h = w/nw, h/nh

            gridS = nw / 13

            xcIndex = int(xc / gridS)
            xcCoor = xc - gridS * xcIndex
            xcCoor = xcCoor / gridS

            ycIndex = int(yc / gridS)
            ycCoor = yc - gridS * ycIndex
            ycCoor = ycCoor /gridS
            if lbl[xcIndex][ycIndex][0] != lbl[xcIndex][ycIndex][1]:
                continue
            lbl[xcIndex][ycIndex] = torch.tensor([c1, c2, 1, 1, xcCoor, ycCoor, w, h, xcCoor, ycCoor, w, h])
            
        return lbl


    def pad_image(self, image, target_size):
        iw, ih = image.size  # 原始图像的尺寸
        w, h = target_size  # 目标图像的尺寸

        scale = min(w / iw, h / ih)  # 转换的最小比例

        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        # image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
        new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
        # // 为整数除法，计算图像的位置
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式

        return new_image


    def squarePadImg(self, img, size):
        ow, oh = img.size
        if ow == oh:
            return img.resize(size)
        padw = False if ow > oh else True
        pad = (ow, ow) if not padw else (oh,oh)
        ig = img
        return self.pad_image(ig,pad).resize(size)


def get_imlist(path):
    return [ os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.jpg') or  f.endswith('.jpeg') or f.endswith('.png')) and os.path.exists(os.path.join(path,  ".".join( f.split(".")[0:-1] ) + ".txt" ) )]

class YOLOLoss(nn.Module):

    def __init__(self, l_coord = 5, l_noobj = 0.5):
        super(YOLOLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def bbox_iou(self, box1, box2, yolo = True):
        if torch.cuda.is_available():
            box1 = box1.cuda()
            box2 = box2.cuda()
        #Get the coordinates of bounding boxes
        if not yolo:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
        else:
            b1_xc, b1_yc, b1_w, b1_h = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
            b2_xc, b2_yc, b2_w, b2_h = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
            
            b1_x1, b1_y1, b1_x2, b1_y2 =  b1_xc - b1_w / 2, b1_yc - b1_h / 2 , b1_xc + b1_w / 2, b1_yc + b1_h / 2
            b2_x1, b2_y1, b2_x2, b2_y2 =  b2_xc - b2_w / 2, b2_yc - b2_h / 2 , b2_xc + b2_w / 2, b2_yc + b2_h / 2


        #get the corrdinates of the intersection rectangle
        inter_rect_x1 =  torch.max(b1_x1, b2_x1)
        inter_rect_y1 =  torch.max(b1_y1, b2_y1)
        inter_rect_x2 =  torch.min(b1_x2, b2_x2)
        inter_rect_y2 =  torch.min(b1_y2, b2_y2)


        if torch.cuda.is_available():
                inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
        else:
                inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

        #Union Area
        b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou

    def forward(self, pred_tensor, target_tensor):
        mseloss = nn.MSELoss(reduction="sum")
        BS = pred_tensor.shape[0]
        biou1 = self.bbox_iou(pred_tensor[...,4:8], target_tensor[...,4:8],  yolo=True)
        biou2 = self.bbox_iou(pred_tensor[...,8:12], target_tensor[...,8:12] , yolo=True)

        target_tensor[...,2] = (biou1 > biou2) * target_tensor[...,2].clone()
        target_tensor[...,3] = (biou1 <= biou2) * target_tensor[...,3].clone()

        target_tensor[...,2] = (biou1) * target_tensor[...,2].clone()
        target_tensor[...,3] = (biou2) * target_tensor[...,3].clone()

        box1mask = target_tensor[...,2] > 0
        box2mask = target_tensor[...,3] > 0

        gridmask = torch.logical_or(target_tensor[..., 2] > 0, target_tensor[..., 3] > 0)

        box1centerloss = mseloss(pred_tensor[...,4:6][box1mask], target_tensor[...,4:6][box1mask])
        box1whloss = mseloss( torch.sqrt(pred_tensor[...,6:8][box1mask] + 1e-6), torch.sqrt(target_tensor[...,6:8][box1mask] + 1e-6) )
        box1closs = mseloss(pred_tensor[...,2][box1mask], target_tensor[...,2][box1mask])
        noobjbox1closs = mseloss(pred_tensor[...,2][~box1mask], target_tensor[...,2][~box1mask])
        
        box1loss = self.l_coord * ( box1centerloss + box1whloss) + box1closs + self.l_noobj * noobjbox1closs


        box2centerloss = mseloss(pred_tensor[...,8:10][box2mask], target_tensor[...,8:10][box2mask])
        box2whloss = mseloss( torch.sqrt(pred_tensor[...,10:12][box2mask] + 1e-6), torch.sqrt(target_tensor[...,10:12][box2mask] + 1e-6) )
        box2closs = mseloss(pred_tensor[...,3][box2mask], target_tensor[...,3][box2mask])
        noobjbox2closs = mseloss(pred_tensor[...,3][~box2mask], target_tensor[...,3][~box2mask])
        
        box2loss = self.l_coord * ( box2centerloss + box2whloss) + box2closs + self.l_noobj * noobjbox2closs

        classloss = mseloss(pred_tensor[...,0:2][gridmask], target_tensor[...,0:2][gridmask])


        loss = (box1loss + box2loss + classloss) / BS

        return loss, (classloss /BS)

def bbox_iou(box1, box2, yolo = False):
    if torch.cuda.is_available():
        box1 = box1.cuda()
        box2 = box2.cuda()
    #Get the coordinates of bounding boxes
    if not yolo:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
    else:
        b1_xc, b1_yc, b1_w, b1_h = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
        b2_xc, b2_yc, b2_w, b2_h = box2[...,0], box2[...,1], box2[...,2], box2[...,3]
        
        b1_x1, b1_y1, b1_x2, b1_y2 =  b1_xc - b1_w / 2, b1_yc - b1_h / 2 , b1_xc + b1_w / 2, b1_yc + b1_h / 2
        b2_x1, b2_y1, b2_x2, b2_y2 =  b2_xc - b2_w / 2, b2_yc - b2_h / 2 , b2_xc + b2_w / 2, b2_yc + b2_h / 2


    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou



