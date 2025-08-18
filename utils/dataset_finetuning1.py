import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torchvision import transforms

class CusImageDataset(Dataset):
    def __init__(self, csv_file, data_path, transform=None):
        """
        Args:
            csv_file (string): CSV 文件的路径，包含图像文件名和标签。
            data_path (string): 图像数据的根目录。
            transform (callable, optional): 可选的变换操作，应用于图像。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_path = data_path
        self.transform = transform
        # 如果没有提供 transform，则使用默认的转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # 打开图像并转换为 RGB
        label = self.data_frame.iloc[idx, 1]  # 标签是字符串类型

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label