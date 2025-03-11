import glob
import os

import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))

        self.filepaths = []
        self.labels = []

        for class_index, class_name in enumerate(self.classes):
            class_folder = os.path.join(data_dir, class_name)
            image_files = glob.glob(os.path.join(class_folder, '*.jpg')) + glob.glob(os.path.join(class_folder, '*.png'))
            for filename in image_files:
                self.filepaths.append(filename)
                self.labels.append(class_index)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img_path = self.filepaths[index]
        label = self.labels[index]

        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, label

## 최적의 가중치로 학습된 네트워크(가중치) 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location='cpu')

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

