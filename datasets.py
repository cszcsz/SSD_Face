import torch
from torch.utils.data import Dataset
import json
import os 
from PIL import Image
from utils import transform

class WiderFaceDataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split.upper()

        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.data_folder = data_folder

        # 读取json数据文件
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as f:
            self.images = json.load(f)
        
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as f:
            self.objects = json.load(f)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # 读取图像
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # 读取该图像中的总人脸数、包围盒、标签
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])    # (n_faces, 4)
        labels = torch.LongTensor(objects['labels'])   # (n_faces)

        # 图像预处理
        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels

    def collate_fn(self, batch):

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        
        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 2 lists of N tensors each
