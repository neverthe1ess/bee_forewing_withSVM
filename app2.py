import torch
import numpy as np
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 0. U-Net, Dataset 등 불러오기
from model import UNet
from util import load, Dataset
from ultralytics import YOLO

yolo_model_path = './saved_models/best.pt'
yolo_model = YOLO(yolo_model_path)

# 1. 모델 로드 및 평가 모드 전환
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

ckpt_dir = './checkpoint'
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
net.eval()

# 2. 데이터셋 / 데이터 로더 정의
data_dir = './dataset'
batch_size = 4

input_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

dataset_test = Dataset(data_dir=data_dir,
                       yolo_model=yolo_model,
                       transform=input_transforms,
                       conf_threshold=0.3)
loader_test = DataLoader(dataset_test, batch_size=batch_size,
                         shuffle=False, num_workers=0)

# 3. 랜드마크 픽셀 군집화 함수
def extract_landmarks_kmeans(binary_mask, K=19):
    """
    binary_mask: (H, W) 0/1 이진 마스크 (numpy)
    K: 찾고자 하는 군집(랜드마크) 개수
    """
    coords = np.argwhere(binary_mask)  # shape=(N, 2)
    num_pixels = coords.shape[0]
    if num_pixels < K:
        return None
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=15)
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)  # (K, 2) 정수 좌표 (y, x)
    return centers


# 4. 테스트 셋 전체 추론 -> 랜드마크 (19개) 추출
landmark_list = []
label_list = []

with torch.no_grad():
    for imgs, labels in loader_test:
        imgs = imgs.to(device)
        outputs = net(imgs)
        
        # threshold=0.5로 2진화
        binarized = (outputs > 0.5).float()  # shape=(B,1,H,W)
        
        # 배치 내 각 이미지를 개별 처리
        for i in range(binarized.shape[0]):
            mask_2d = binarized[i, 0].cpu().numpy() # shape =(H,W)
            
            # 회전 정렬 (필요하다면 사용)
            # 여러 이미지를 같은 각도로 보정하려는 용도
            # rotated = shift_angle([mask_2d])[0]

            # K-Means로 19개 중심점(랜드마크) 찾기
            centers = extract_landmarks_kmeans(mask_2d, K=19)
            if centers is not None:
                landmark_list.append(centers)
                label_list.append(labels[i].item())
            else:
                # 활성화 픽셀이 19개 미만 -> landmark 추출 불가
                print("활성화 픽셀이 19개 미만: 스킵합니다.")

all_dots = np.array(landmark_list, dtype=float)  # (N,19,2)
label_array = np.array(label_list)               # (N,)


# 5. 랜드마크 정렬 함수
def sort_dots(dots, x_thresh=5):
    """
    dots: (N, 2) 형태의 랜드마크 좌표 (y, x)
    x좌표를 전역 오름차순으로 본 뒤,
    x좌표 차이가 x_thresh 이하인 점들은 같은 '묶음'으로 보고,
    그 묶음 안에서는 y 오름차순으로 정렬합니다.
    """
    # 1) x좌표로 전체 오름차순 정렬
    #    dots[:,1]이 x, dots[:,0]이 y
    sorted_by_x = dots[np.argsort(dots[:, 1])]
    
    # 2) 그룹핑 & 그룹 내 y 오름차순
    groups = []
    current_group = [sorted_by_x[0]]

    for i in range(1, len(sorted_by_x)):
        # 현재 보고 있는 점
        point = sorted_by_x[i]
        # 현재 그룹의 마지막 점
        last_point_in_group = current_group[-1]
        
        # 만약 x좌표 차이가 x_thresh 이하면 같은 그룹에 계속 쌓는다
        if (point[1] - last_point_in_group[1]) <= x_thresh:
            current_group.append(point)
        else:
            # 기존 그룹 확정 → 그룹 내 y 오름차순 정렬
            current_group = np.array(current_group)
            current_group = current_group[np.argsort(current_group[:, 0])]
            groups.append(current_group)

            # 새로운 그룹 시작
            current_group = [point]

    # 마지막 그룹도 정리
    current_group = np.array(current_group)
    current_group = current_group[np.argsort(current_group[:, 0])]
    groups.append(current_group)

    # 3) 모든 그룹을 순서대로 합침
    result = np.concatenate(groups, axis=0)
    result = adjust_element(result)
    return result

# 5-2. 정교하게 한번 더 정렬
def adjust_element(element):
    """
    element: 랜드마크 점 집합
    y값 우선으로 한번 더 정렬
    """
    element[5:10] = element[5:10][np.argsort(element[5:10, 0])]
    e1 = element[10]
    e2 = element[11]
    if e1[0] < e2[0]:
        element[10] = e2
        element[11] = e1
    return element


def calculate_angles_between_nearest_points(dots):
    """
    dots: (19, 2) (y, x) 좌표
    각 점마다 미리 정해진 다른 점과 연결하여,
    두 점을 잇는 선과 x축 사이의 각도(도 단위)를 계산.

    반환: (nearest_points, angles)
      nearest_points: (19, 2)
      angles: (19,)  # 각 점에서 지정된 점과의 각도(degree)
    """
    n_points = dots.shape[0]
    nearest_pts = []
    angles = []

    nearest_list = [2, 0, 3, 4, 5, 12, 13, 14, 9, 15, 5, 6, 13, 14, 16, 17, 17, 18, 1] 

    for i in range(n_points):
        current = dots[i]
        nearest_pt = dots[nearest_list[i]]
        
        dy, dx = nearest_pt - current
        angle_rad = np.arctan2(-dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        nearest_pts.append(nearest_pt)
        angles.append(angle_deg)
    
    return np.array(nearest_pts), np.array(angles)



def calculate_angles(dots):
    """
    A1 -> 14 / 17  & 17 / 16 
    A4 -> 14 / 17  & 13 / 17
    B3 -> 14 / 15  & 14 / 17
    B4 -> 13 / 14  & 14 / 17
    D7 -> 14 / 15  & 9  / 15
    G7 -> 9  / 14  & 9  / 15
    G18 -> 1 / 9   & 8  / 9
    """

    list_a = [14, 14, 14, 13, 14, 9, 1]
    list_b = [17, 17, 15, 14, 15, 14, 9]
    list_c = [17, 13, 14, 14, 9, 9, 8]
    list_d = [16, 17, 17, 17, 15, 15, 9]

    angles = []

    for i in range(len(list_a)):
        A = dots[list_a[i]]
        B = dots[list_b[i]]
        C = dots[list_c[i]]
        D = dots[list_d[i]]

        u = np.array([B[1] - A[1], B[0] - A[0]])  # [Δy, Δx]
        v = np.array([D[1] - C[1], D[0] - C[0]])  # [Δy, Δx]

        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        # 각도 계산 (라디안)
        angle_rad = np.arccos(dot_product / (norm_u * norm_v))

        # 도(degree)로 변환
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

    return np.array(angles)



# 6-3 Centroid Size 구하기
def calculate_centroid_size(dots):
    '''
    dots: (19, 2) (y, x) 좌표

    19개의 점의 중심점인 centroid point를 구한 뒤 
    centroid point와 19개 점 사이의 거리의 평균인 
    centroid size 계산

    반환: centroid_size
    '''
    n_points = dots.shape[0]
    centroid_point = np.sum(dots, axis=0) / n_points
    distances = np.sqrt(np.sum((dots - centroid_point)**2, axis=1))
    centroid_size = np.mean(distances)
    return centroid_size
    

# 7. 각도 -> sin/cos 변환, -180~180 -> 0~360
def convert_angles_to_sin_cos(angles_deg):
    """
    angles_deg: shape=(19,)  # degree
    각도 1개에 대해 [sin, cos] 2개로 확장 -> 최종 38차원
    """
    sin_cos_list = []
    for angle in angles_deg:
        rad = np.deg2rad(angle)  # deg -> rad
        sin_cos_list.append(np.sin(rad))
        sin_cos_list.append(np.cos(rad))
    return sin_cos_list


# 8. 모든 샘플에 대해 19개 각도 -> sin/cos(38차원) 피처 생성
angle_features_sin_cos = []
degree_list = []

for i in range(len(all_dots)):
    # (19,2) 정렬 (원하는 기준으로)
    sorted_landmarks = sort_dots(all_dots[i])

    # 각도 계산
    _, angles = calculate_angles_between_nearest_points(sorted_landmarks)
    non_angles = calculate_angles(sorted_landmarks)

    degree_list.append(np.concatenate((angles, non_angles)))
    

    # sin/cos 변환
    feature_38 = np.array(convert_angles_to_sin_cos(angles))
    centroid_size = calculate_centroid_size(sorted_landmarks)
    feature_39 = np.append(feature_38, centroid_size)  # numpy array로 처리
    angle_features_sin_cos.append(feature_39)


for i in range(len(dataset_test)):
    _, label = dataset_test[i]
    print(degree_list[i])
    print(label)

X_sin_cos = np.array(angle_features_sin_cos)  # shape: (N, 38)
y = label_array  # (N,)


# 9. SVM 분류
# 9-1. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sin_cos, y, test_size=0.2, random_state=16, stratify=y
)

# 9-2. (선택) 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 9-3. SVM 학습
param_grid = {
    'C': [0.1, 1, 10, 30, 100],
    'gamma': [0.01, 0.1, 0.3, 1, 10]
}

grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), 
                         param_grid, 
                         cv=5, 
                         scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 9-4. 예측/평가
y_pred = best_model.predict(X_test_scaled)
print(y_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"[SVM 분류] 정확도: {acc*100:.2f}%")

# 9-5. 모델 저장
import joblib
import os

# 저장할 디렉토리 생성
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

# 모델 저장
joblib.dump(best_model, os.path.join(save_dir, 'svm_model.joblib'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))

print(f"\n모델이 {save_dir} 디렉토리에 저장되었습니다.")
print("- svm_model.joblib: 학습된 SVM 모델")
print("- scaler.joblib: 데이터 스케일링 모델")




def visualize_overlapped_landmarks(dataset, all_centers, sample_idx1=0, sample_idx2=80, alpha1=0.7, alpha2=0.3):
    """
    dataset: Dataset 객체
    all_centers: (N,19,2) 형태
    sample_idx1, sample_idx2: 비교할 두 샘플의 인덱스
    alpha1, alpha2: 각 이미지의 투명도 (0~1)
    """
    def prepare_image(img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            else:
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.array(img)
        return img_np

    # 두 샘플 이미지 준비
    img1, label1 = dataset[sample_idx1]
    img2, label2 = dataset[sample_idx2]
    
    img1_np = prepare_image(img1)
    img2_np = prepare_image(img2)
    
    # 두 샘플의 랜드마크 준비
    centers1 = sort_dots(all_centers[sample_idx1])
    centers2 = sort_dots(all_centers[sample_idx2])

    plt.figure(figsize=(10, 8))
    
    # 첫 번째 이미지 (빨간색 계열)
    plt.imshow(img1_np, cmap='gray', alpha=alpha1)
    # if centers1 is not None:
    #     plt.scatter(centers1[:,1], centers1[:,0], s=30, marker='x', c='red', label=f'Sample {sample_idx1}')
    #     for idx, (y, x) in enumerate(centers1):
    #         plt.annotate(str(idx), (x, y), xytext=(5, 5), textcoords='offset points', 
    #                     fontsize=16, color='red')
    
    # 두 번째 이미지 (파란색 계열)
    # plt.imshow(img2_np, cmap='gray', alpha=alpha2)
    # if centers2 is not None:
    #     plt.scatter(centers2[:,1], centers2[:,0], s=30, marker='x', c='blue', label=f'Sample {sample_idx2}')
    #     for idx, (y, x) in enumerate(centers2):
    #         plt.annotate(str(idx), (x, y), xytext=(-5, -5), textcoords='offset points', 
    #                     fontsize=16, color='blue')
    
    plt.title(f"Overlapped Samples\nRed: Sample {sample_idx1} (Label={label1})\nBlue: Sample {sample_idx2} (Label={label2})")
    plt.legend()
    plt.show()


# 시각화 예시 (0번과 1번 샘플을 오버랩)
for i in range(84):
    visualize_overlapped_landmarks(dataset_test, all_dots, sample_idx1=0, sample_idx2=i+1, alpha1=0.7, alpha2=0.3)
