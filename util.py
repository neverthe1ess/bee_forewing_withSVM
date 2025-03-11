import glob
import os
import cv2 
import numpy as np
import torch

from PIL import Image
from sklearn.decomposition import PCA

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

# 기울어진 이미지를 정렬(PCA)
def shift_angle(masks):
    '''
    input -> 2차원 넘파이 배열(rows, cols)이 담긴 리스트
    '''
    for index, img in enumerate(masks):
        rows, cols = img.shape
        center = (cols/2,rows/2)
        _, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)

        dots = np.argwhere(img == 255) # 흰색 찾기
        if len(dots) == 0:
            continue

        pca = PCA(n_components=2)
        pca.fit_transform(dots)

        vec = pca.components_ # eigen vector
        angle = np.degrees(np.arctan2(*vec.T[::-1])) % 360.0
		
		# 이미지 이동(평균 좌표가 이미지의 중심점에 오도록 평행 이동)
        M = np.float32([[1,0,-(pca.mean_[1]-center[1])],
                        [0,1,-(pca.mean_[0]-center[0])]
                        ])
        img = cv2.warpAffine(img,M,(cols,rows))
		
		# 계산된 PCA 각도만큼 회전 
        M = cv2.getRotationMatrix2D((center),-angle[1],1)
        img = cv2.warpAffine(img,M,(cols,rows))
        
        masks[index] = img
    
    return masks
