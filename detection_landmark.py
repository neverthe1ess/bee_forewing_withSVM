import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from eval import output
from model import UNet
from util import load, Dataset


# 1. 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr = 1e-3)

ckpt_dir = './checkpoint'
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net = net, optim=optim)
net.eval()

# 2. 데이터셋 / 데이터 로더 정의
batch_size=4
data_dir = './dataset'

input_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

dataset_test = Dataset(data_dir='./dataset',
                         transform=input_transforms)
loader_test = DataLoader(dataset_test, batch_size=batch_size,
                         shuffle=False, num_workers=0)

# 3. 랜드마크 마스크에서 랜드마크 추출
def extract_landmarks(output):
    binary_output = (torch.sigmoid(output) > 0.5).cpu().numpy()  # (배치, 1, H, W) 아직 텐서임

    landmarks = []
    for i in range(binary_output.shape[0]):  # 배치 내부에서 0~배치 사이즈 샘플 반복
        # 흰색인 픽셀(=마스크가 활성화된 픽셀)의 좌표 인덱스
        indices = np.argwhere(binary_output[i, 0])  # shape: (?, 2)

        # 여기서는 19개 픽셀이 있다고 가정
        if len(indices) == 19:
            landmarks.append(indices)
        else:
            # 19개가 아니면 None 처리하거나, 스킵할 수도 있음
            landmarks.append(None)

    # 최종적으로 (배치 사이즈, 19, 2) 형태(또는 None)로 관리
    return landmarks

# 4. 추론
landmark_list = []
label_list = []


with torch.no_grad():
    for imgs, labels in loader_test: # 배치 만큼 Iterative
        imgs = imgs.to(device)
        outputs = net(imgs)

        # 랜드마크 추출
        landmark_batch = extract_landmarks(outputs)

        for landmark, label in zip(landmark_batch, labels):
            if landmark is not None:
                landmark_list.append(landmark)
                label_list.append(label.item())

landmark_list = np.array(landmark_list)
label_list = np.array(label_list)

print("추론 완료! 랜드마크 개수:", len(landmark_list))
print("라벨 개수:", len(label_list))

