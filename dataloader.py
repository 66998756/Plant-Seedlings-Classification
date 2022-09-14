import os
from os import listdir
import glob
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

img_label_dict = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
                  "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize", 
                  "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                  "Sugar beet"]


class remove_background():
    def __call__(self, img):
        img = np.array(img)
        blurImg = cv2.GaussianBlur(img, (5, 5), 0)
        hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_RGB2HSV)
        
        lower_green = (25, 40, 50)
        upper_green = (75, 255, 255)
        mask = cv2.inRange(hsvImg, lower_green, upper_green) 
        
        bMask = mask > 0 
        # Apply the mask
        clear = np.zeros_like(img, np.uint8)  # Create empty image
        clear[bMask] = img[bMask]

        return Image.fromarray(clear)


class test_img():
    def __call__(self, img):
        img.save("./tmp.jpg")
        return img

class prob_resize_crop():
    def __init__(self, img_resize, p=0.5):
        self.img_resize = img_resize
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            img = transforms.RandomResizedCrop((self.img_resize, self.img_resize))(img)
            return img
        return img


class PlantDataset(Dataset):
    def __init__(self, label_file, mode, class_num=12, img_resize=[192, 192], transform=None):
        self.transform = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize([int(img_resize+img_resize/4), int(img_resize+img_resize/4)]), 
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                                 transforms.RandomRotation([-45, 45], expand=False, center=(int((img_resize + img_resize/4)/2), int((img_resize + img_resize/4)/2))),
                                 # prob_resize_crop(p=0.5, img_resize=img_resize),
                                 # transforms.RandomResizedCrop((img_resize, img_resize)),
                                 # transforms.CenterCrop([img_resize, img_resize]),
                                 transforms.Resize([img_resize, img_resize]), 
                                 remove_background(),
                                 # test_img(),
                                 transforms.ToTensor(),
                                 # transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=(255, 255, 255)),
                                 transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]),
                             ])
        self.test_transform = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize([img_resize, img_resize]), 
                                 remove_background(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]),
                             ])
        
        self.mode = mode
        self.class_num = class_num
        self.img_csv = pd.read_csv(label_file)
        self.img_names = self.img_csv['file']
        self.img_labels = self.img_csv['species']
        
        # self.data_dict = {}
        # for label in labels:
        #     datas = listdir(os.path.join(path, label))
        #     for data in datas:
        #         if label in self.data_dict.keys():
        #             self.data_dict[label].append(data)
        #         else:
        #             self.data_dict[label] = []
    
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = os.path.join(
            "./dataset",
            self.mode,
            self.img_labels[item],
            self.img_names[item],
        )
        
        img_label = [0 for i in range(12)]
        img_label_ = self.img_labels[item]
        img_label[img_label_dict.index(img_label_)] += 1
        img_label = torch.FloatTensor(img_label)

        img = cv2.imread(img_path)
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # img = cv2.imdecode(np.fromfile(img_path, dtype='float64'), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.mode_ == "train":
            img = self.transform(img)
        else:
            img = self.test_transform(img)
        return img, img_label

    def set_mode(self, mode_):
        self.mode_ = mode_


if __name__ == "__main__":
    testDataset = PlantDataset("train_labels.csv", "train")
    for i in range(len(testDataset)):
        print(testDataset[i])
        input()
    
