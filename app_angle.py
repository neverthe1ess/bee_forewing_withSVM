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
    dots: (19, 2) 형태의 랜드마크 좌표 배열
    -> x좌표(dots[:, 1])를 기준으로 오름차순 정렬하여 반환
    """
    # x좌표(첫 번째 열)를 기준으로 오름차순 정렬
    sorted_indices = np.argsort(dots[:, 1])
    sorted_dots = dots[sorted_indices]
    return sorted_dots

# 4.3 각도 계산
def calculate_angles_between_nearest_points(dots):
    """
    dots: (19, 2) 형태의 랜드마크 좌표 배열
    -> 각 점에서 가장 가까운 다른 점을 찾고, 두 점을 잇는 선과 x축 사이의 각도를 계산
    """
    n_points = dots.shape[0]  # 점의 개수 (19)
    nearest_points = []
    angles = []
    
    for i in range(n_points):
        current_point = dots[i]
        
        # 자기 자신을 제외한 모든 점과의 거리 계산
        distances = np.sqrt(np.sum((dots - current_point)**2, axis=1))
        
        # 자기 자신과의 거리는 0이므로, 가장 작은 값이 아닌 두 번째로 작은 값을 찾아야 함
        # inf로 설정하여 최소값 계산에서 제외
        distances[i] = np.inf
        
        # 가장 가까운 점의 인덱스 찾기
        nearest_idx = np.argmin(distances)
        nearest_point = dots[nearest_idx]
        
        # 두 점 사이의 벡터 계산 (y, x)
        vector = nearest_point - current_point
        
        # x축과의 각도 계산 (라디안)
        # arctan2는 y, x 순서로 인자를 받음
        angle_rad = np.arctan2(vector[0], vector[1])
        
        # 라디안을 도(degree)로 변환
        angle_deg = np.degrees(angle_rad)
        
        nearest_points.append(nearest_point)
        angles.append(angle_deg)
    
    return np.array(nearest_points), np.array(angles)

# 모든 샘플에 대해 각 점의 가장 가까운 점과 각도 계산
for i in range(len(all_dots)):
    sorted_landmarks = sort_dots(all_dots[i])
    nearest_points, angles = calculate_angles_between_nearest_points(sorted_landmarks)
    
    print(f"Sample {i}:")
    for j in range(len(sorted_landmarks)):
        print(f"  점 {j} {sorted_landmarks[j]}: 가장 가까운 점 {nearest_points[j]}, 각도 {angles[j]:.2f}도")



# 5. Procrustes 정규화
def procrustes_transform(X):
    X_transformed = []
    ref_shape = X[0].reshape(19, 2)
    for shape in X:
        _, transformed, _ = procrustes(ref_shape, shape)
        X_transformed.append(transformed.flatten())
    return np.array(X_transformed)


X = procrustes_transform(all_dots)
print(sort_dots(all_dots[0]))
y = label_array

# 6. SVM 분류
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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





