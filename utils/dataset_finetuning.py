import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms


class CusImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        # 增加额外的数据增强操作
        self.extra_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # 增加旋转操作
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.df = pd.read_csv(csv_file)
        self.data_path = data_path
        self.filenames, self.labels = self.df['Image'], self.df['Label']
        self.label_list = list(set(self.df['Label'].values.tolist()))
        self.label_list.sort()
        # 用于记录每个类别的样本数量，初始化为 0
        self.class_count = {label: 0 for label in self.label_list}
        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except Exception as e:
            print("img_path ======== {}".format(img_path))
            print(e)
        if transform is not None:
            img = transform(img)
        return img

    def __getitem__(self, index):
        key = self.filenames[index]
        image_path = os.path.join(self.data_path, key)
        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]
        labels = self.label_list.index(labels)
        labels = np.array(labels)
        self.class_count[self.label_list[labels]] += 1
        if self.class_count[self.label_list[labels]] < 2000:
            imgs = self.get_imgs(image_path, self.extra_transform)  # 对数据量小的类别使用额外的数据增强
        return imgs, labels, key

    def __len__(self):
        return len(self.filenames)


class RetrivalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            self.image_paths += [(os.path.join(category_path, f), category) for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.class_count = {category: 0 for category in self.categories}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, category = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        self.class_count[category] += 1
        return image, img_path, category


