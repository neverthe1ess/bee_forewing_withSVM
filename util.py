import glob
import os
import cv2 
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from ultralytics import YOLO

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, yolo_model, transform=None, conf_threshold=0.25):
        super().__init__()
        self.data_dir = data_dir
        self.yolo_model = yolo_model
        self.transform = transform
        self.conf_threshold = conf_threshold
        self.classes = sorted(os.listdir(data_dir))

        self.filepaths = []
        self.labels = []

        for class_index, class_name in enumerate(self.classes):
            class_folder = os.path.join(data_dir, class_name)
            image_files = glob.glob(os.path.join(class_folder, '*.jpg')) \
                        + glob.glob(os.path.join(class_folder, '*.png'))
            for filename in image_files:
                self.filepaths.append(filename)
                self.labels.append(class_index)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img_path = self.filepaths[index]
        label = self.labels[index]

        img = Image.open(img_path).convert("L")

        # ------------------------------------------------------------
        # 3) YOLO 모델로 추론 -> bounding box 얻기
        # ------------------------------------------------------------
        results = self.yolo_model.predict(img, verbose=False)  
        # 결과는 리스트이고, 보통 results[0]에 해당 이미지의 예측 정보가 들어있음.
        # YOLOv8에서는 results[0].boxes 내부에 여러 box가 있을 수 있음.

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes  # ultralytics.yolo.engine.results.Boxes 객체
            best_box = None
            best_conf = 0.0

            for box in boxes:
                conf = float(box.conf[0])  # YOLOv8 box.conf는 tensor, 여기선 첫 번째 값만.
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()  # 텐서를 numpy로 변환 (xmin, ymin, xmax, ymax)

            if best_box is not None and best_conf >= self.conf_threshold:
                xmin, ymin, xmax, ymax = best_box
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

                # pillow에서 crop은 (left, upper, right, lower)
                img = img.crop((xmin, ymin, xmax, ymax))
        else:
            # 만약 검출된 box가 전혀 없다면, 원본 이미지를 그대로 사용하거나 예외 처리
            pass

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

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

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