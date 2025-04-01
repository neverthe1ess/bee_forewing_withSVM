import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# 모델/데이터 로드 및 저장용
import joblib
import os

# 0. U-Net, Dataset 등 불러오기
from model import UNet
from util import load, Dataset
from ultralytics import YOLO

# YOLO 모델 불러오기
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

# 3. 랜드마크 픽셀 군집화 함수 (K-Means)
def extract_landmarks_kmeans(binary_mask, K=19):
    """
    binary_mask: (H, W) 0/1 이진 마스크 (numpy)
    K: 찾고자 하는 군집(랜드마크) 개수
    """
    coords = np.argwhere(binary_mask)  # shape=(N, 2)  (y, x) 좌표
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
            mask_2d = binarized[i, 0].cpu().numpy()  # shape =(H,W)

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

# 5. 랜드마크 정렬 함수들
def sort_dots(dots, x_thresh=5):
    """
    dots: (N,2) 형태의 랜드마크 좌표 (y, x)
    x좌표를 기준으로 오름차순 정렬 후, 근접 x끼리는 묶어서 y오름차순 정렬
    """
    # (1) x좌표로 전체 오름차순 정렬
    sorted_by_x = dots[np.argsort(dots[:, 1])]
    
    # (2) 그룹핑 & 그룹 내 y 오름차순
    groups = []
    current_group = [sorted_by_x[0]]

    for i in range(1, len(sorted_by_x)):
        point = sorted_by_x[i]
        last_point_in_group = current_group[-1]
        
        # x좌표 차이가 x_thresh 이하 -> 같은 그룹으로 취급
        if (point[1] - last_point_in_group[1]) <= x_thresh:
            current_group.append(point)
        else:
            # 기존 그룹 확정 (y 오름차순 정렬 후 저장)
            current_group = np.array(current_group)
            current_group = current_group[np.argsort(current_group[:, 0])]
            groups.append(current_group)

            # 새 그룹 시작
            current_group = [point]

    # 마지막 그룹도 정리
    current_group = np.array(current_group)
    current_group = current_group[np.argsort(current_group[:, 0])]
    groups.append(current_group)

    # (3) 모든 그룹을 순서대로 합침
    result = np.concatenate(groups, axis=0)
    result = adjust_element(result)
    return result

def adjust_element(element):
    """
    element: (19,2) 형태의 랜드마크
    y값 우선 재정렬 등 필요하면 커스터마이징
    """
    # 일괄 정렬 예시 - 사용자에 맞춰 수정 가능
    element[5:10] = element[5:10][np.argsort(element[5:10, 0])]

    e1 = element[10]
    e2 = element[11]
    # 간단한 예시로, 10번/11번의 y좌표를 비교 후 스왑
    if e1[0] < e2[0]:
        element[10] = e2
        element[11] = e1
    return element


# 6. 각도 계산 함수들
def calculate_angles_between_nearest_points(dots):
    """
    dots: (19, 2) (y, x)
    특정 규칙(nearest_list)에 따라 연결된 두 점(현재점-연결점) 사이 각도 계산
    """
    n_points = dots.shape[0]
    nearest_list = [2, 0, 3, 4, 5, 12, 13, 14, 9, 15, 5, 6, 13, 14, 16, 17, 17, 18, 1]

    nearest_pts = []
    angles = []

    for i in range(n_points):
        current = dots[i]
        nearest_pt = dots[nearest_list[i]]
        
        dy, dx = nearest_pt - current
        angle_rad = np.arctan2(-dy, dx)  # -dy 주의 (사용자 정의)
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

    (사용자 정의 파트)
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

        u = np.array([B[1] - A[1], B[0] - A[0]])  # (dx, dy)
        v = np.array([D[1] - C[1], D[0] - C[0]])  # (dx, dy)

        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        angle_rad = np.arccos(dot_product / (norm_u * norm_v))
        angle_deg = np.degrees(angle_rad)
        angles.append(angle_deg)

    return np.array(angles)


# 7. Centroid Size 구하기
def calculate_centroid_size(dots):
    """
    dots: (19, 2) (y,x)
    19개 점의 중심(centroid)과 각 점의 거리 -> 평균
    """
    n_points = dots.shape[0]
    centroid_point = np.sum(dots, axis=0) / n_points
    distances = np.sqrt(np.sum((dots - centroid_point)**2, axis=1))
    centroid_size = np.mean(distances)
    return centroid_size

# 7-2. 각도 -> sin/cos 변환
def convert_angles_to_sin_cos(angles_deg):
    """
    angles_deg: shape=(19,)  # degree
    각도 1개당 sin, cos -> 최종 38차원
    """
    sin_cos_list = []
    for angle in angles_deg:
        rad = np.deg2rad(angle)
        sin_cos_list.append(np.sin(rad))
        sin_cos_list.append(np.cos(rad))
    return sin_cos_list

# 8. 모든 샘플에 대해 각도 + sin/cos(38차원) + centroid_size(1차원) => 총 39차원
angle_features_sin_cos = []
degree_list = []

for i in range(len(all_dots)):
    # (19,2) 정렬
    sorted_landmarks = sort_dots(all_dots[i])

    # (a) 연결된 점들 사이 각도
    _, angles = calculate_angles_between_nearest_points(sorted_landmarks)
    # (b) 사용자 정의 각도
    non_angles = calculate_angles(sorted_landmarks)
    
    degree_list.append(np.concatenate((angles, non_angles)))

    # sin/cos 변환
    feature_38 = np.array(convert_angles_to_sin_cos(angles))
    centroid_size = calculate_centroid_size(sorted_landmarks)
    feature_39 = np.append(feature_38, centroid_size)  # [38 + 1]
    angle_features_sin_cos.append(feature_39)

X_sin_cos = np.array(angle_features_sin_cos)  # shape: (N,39)
y = label_array                              # shape: (N,)


# =================================================
# 9. LDA 분류
# =================================================
# 9-1. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sin_cos, y, test_size=0.2, random_state=16, stratify=y
)

# 9-2. (선택) 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 9-3. LDA 학습
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train)

# 9-4. 예측/평가
y_pred = lda_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("[테스트셋 실제 라벨]", y_test)
print("[테스트셋 예측 라벨]", y_pred)
print(f"[LDA 분류] 정확도: {acc*100:.2f}%")

# 9-5. 모델 저장
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

joblib.dump(lda_model, os.path.join(save_dir, 'lda_model.joblib'))
joblib.dump(scaler,    os.path.join(save_dir, 'lda_scaler.joblib'))

print(f"\n[LDA] 모델이 {save_dir} 디렉토리에 저장되었습니다.")
print("- lda_model.joblib: 학습된 LDA 모델")
print("- lda_scaler.joblib: 데이터 스케일링 모델")


# =================================================
# 10. 시각화 함수
# =================================================
def visualize_overlapped_landmarks(dataset, all_centers, sample_idx1=0, sample_idx2=80, alpha1=0.7, alpha2=0.3):
    """
    dataset: Dataset 객체
    all_centers: (N,19,2)
    sample_idx1, sample_idx2: 비교할 샘플 인덱스
    alpha1, alpha2: 이미지 투명도
    """
    def prepare_image(img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
            # (C,H,W) -> (H,W) 또는 (H,W,C)
            if img_np.shape[0] == 1:
                # 흑백
                img_np = img_np[0]
            else:
                # RGB일 경우
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            img_np = np.array(img)
        return img_np

    # 두 샘플 이미지
    img1, label1 = dataset[sample_idx1]
    img2, label2 = dataset[sample_idx2]

    img1_np = prepare_image(img1)
    img2_np = prepare_image(img2)

    # 두 샘플의 랜드마크
    centers1 = all_centers[sample_idx1]
    centers2 = all_centers[sample_idx2]

    plt.figure(figsize=(10, 8))
    
    # 첫 번째 이미지
    plt.imshow(img1_np, cmap='gray', alpha=alpha1)
    if centers1 is not None:
        plt.scatter(centers1[:,1], centers1[:,0], s=30, marker='x', c='red', label=f'Sample {sample_idx1}')
        for idx, (y, x) in enumerate(centers1):
            plt.annotate(str(idx), (x, y), xytext=(5, 5), textcoords='offset points', 
                         fontsize=10, color='red')
    
    # 두 번째 이미지 오버레이
    plt.imshow(img2_np, cmap='gray', alpha=alpha2)
    if centers2 is not None:
        plt.scatter(centers2[:,1], centers2[:,0], s=30, marker='x', c='blue', label=f'Sample {sample_idx2}')
        for idx, (y, x) in enumerate(centers2):
            plt.annotate(str(idx), (x, y), xytext=(-5, -5), textcoords='offset points', 
                         fontsize=10, color='blue')
    
    plt.title(f"Overlapped Samples\nRed: Sample {sample_idx1} (Label={label1}) | Blue: Sample {sample_idx2} (Label={label2})")
    plt.legend()
    plt.show()


# 예시: 0번 샘플과 1~84번 샘플 비교
for i in range(84):
    visualize_overlapped_landmarks(dataset_test, all_dots, sample_idx1=0, sample_idx2=i+1)