import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import procrustes as pr 

from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from model import UNet
from util import load, Dataset, shift_angle

# 1. 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

ckpt_dir = './checkpoint'
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
net.eval()

# 2. 데이터셋 / 데이터 로더 정의
batch_size = 4
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

# 3-1. 랜드마크 픽셀 군집화 함수
def extract_landmarks_kmeans(binary_mask, K=19):
    coords = np.argwhere(binary_mask)  # (N, 2)
    num_pixels = coords.shape[0]
    if num_pixels < K:
        return None

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)  # (K, 2) 정수 좌표 (y, x)
    return centers


# 4. 추론 (U-Net → K-means 랜드마크 추출)
landmark_list = []
label_list = []

with torch.no_grad():
    # 배치 단위
    for imgs, labels in loader_test:
        # imgs.shape = (Batch_size, 1, 512, 512)
        imgs = imgs.to(device)
        outputs = net(imgs)
        binarized = (outputs > 0.5).float() # 2진화 Tensor

        for i in range(binarized.shape[0]): # Batch_size 만큼 반복
            # (1,C, H, W) -> (H, W)
            mask_2d = binarized[i, 0].cpu().numpy()

            # 회전 정렬
            rotated = shift_angle([mask_2d])[0]

            # 뭉친 픽셀 들 사이에서 중심 점을 찾기 (K-Means)
            centers = extract_landmarks_kmeans(rotated, K=19)
            if centers is not None:
                landmark_list.append(centers)
                label_list.append(labels[i].item())
            else:
                print("활성화 픽셀이 19개 미만임")

# landmark_list -> (N, 19, 2) 
all_dots = np.array(landmark_list, dtype=float)
label_array = np.array(label_list)


# 4-2. 좌표 정렬
def sort_dots(dots):
    """
    dots: (19, 2) 형태의 랜드마크 좌표 (y, x)
    x좌표 기준 오름차순 정렬
    x좌표 픽셀 차이가 10 이하인 경우 y좌표로 정렬
    """
    # x좌표로 먼저 정렬
    sorted_by_x = dots[np.argsort(dots[:, 1])]
    
    # 결과 배열 초기화
    result = np.zeros_like(sorted_by_x)
    
    # 현재 처리 중인 인덱스(이미 처리된 인덱스 중복 처리 방지)
    current_idx = 0
    
    print(sorted_by_x)

    while current_idx < len(sorted_by_x):
        # 현재 x값
        current_x = sorted_by_x[current_idx, 1]
        
        # x값이 현재값과 10 이하 차이나는 점들의 인덱스 찾기
        similar_x_indices = []
        for i in range(current_idx, len(sorted_by_x)):
            if sorted_by_x[i, 1] - current_x <= 10:
                similar_x_indices.append(i)
            else:
                break
        
        # 해당 점들을 y값으로 정렬(자기 자신을 포함하기 때문에 1 초과)
        if len(similar_x_indices) > 1:
            points_to_sort = sorted_by_x[similar_x_indices]
            sorted_by_y = points_to_sort[np.argsort(points_to_sort[:, 0])]
            result[current_idx:current_idx+len(similar_x_indices)] = sorted_by_y
        else:
            result[current_idx] = sorted_by_x[current_idx]
        
        # 다음 처리할 인덱스로 이동
        current_idx += len(similar_x_indices)
    
    return result


# 5. 정렬된 좌표 (Aligned Coordinates)
def procrustes_transform(X):
    """
    Orthogonal Procrustes Analysis를 이용하여 좌표 정렬
    - 위치(translation), 크기(scale), 회전(rotation) 불변성 보장
    
    X: (N, 19, 2) 형태의 랜드마크 좌표 배열
    반환: 정렬된 좌표 배열 (N, 38) - 각 좌표가 평탄화됨
    """
    from scipy.spatial import procrustes
    
    X_transformed = []
    # 모든 개체의 랜드마크 좌표 평균을 ref shape으로 사용
    ref_shape = np.mean(X, axis = 0)
    
    for shape in X:
        # procrustes 함수는 (mtx1, mtx2) -> (disparity, mtx1, mtx2) 반환
        # 여기서 mtx2는 mtx1에 최적으로 정렬된 형태
        _, _, transformed = procrustes(ref_shape, shape.reshape(19, 2))
        X_transformed.append(transformed.flatten())  # (19,2) -> (38,) 평탄화 | (Yi, Xi)
    
    return np.array(X_transformed)

dots_list = []

for i in range(len(all_dots)):
    # (19,2) 정렬 (원하는 기준으로)
    sorted_landmarks = sort_dots(all_dots[i])
    dots_list.append(sorted_landmarks)

dots_list = np.array(dots_list)

X = procrustes_transform(dots_list)
y = label_array

# 6. SVM 분류
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)

svm_model = SVC(kernel='rbf', C=30, gamma=0.30, probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM 분류 정확도: {accuracy:.2%}")


# 7. 랜드마크 시각화 함수 (모든 샘플)
def visualize_landmarks_on_image(dataset, all_centers):
    """
    dataset: Dataset 객체
    all_centers: (N, 19, 2) 형태의 전체 랜드마크 (또는 None)
    -> 모든 샘플(0 ~ N-1)에 대해서 이미지+랜드마크를 표시
    """
    n = len(dataset)
    for i in range(1):
        # Dataset에서 (img, label) 가져오기
        img, label = dataset[i]

        # img가 Tensor이면 (C,H,W)이므로 numpy 변환
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()  # (C,H,W)
            if img_np.shape[0] == 1:
                # 그레이스케일 => (H,W)
                img_np = img_np[0]
            else:
                # RGB => (H,W,3)
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            # PIL 이미지
            img_np = np.array(img)

        centers = all_centers[i]  # (19, 2) 또는 None
        plt.figure()
        plt.imshow(img_np, cmap='gray')
        if centers is not None:
            plt.scatter(centers[:, 1], centers[:, 0], s=30, marker='x')
            plt.title(f"Sample {i}, Label={label}")
        else:
            plt.title(f"Sample {i}, Label={label} (No Landmarks)")
        plt.show()


# 8. 모든 샘플 시각화
visualize_landmarks_on_image(dataset_test, all_dots)





