import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull

# 0. U-Net, Dataset 등 불러오기
from model import UNet
from util import load, Dataset


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
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

dataset_test = Dataset(data_dir=data_dir,
                       transform=input_transforms)
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
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
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
            
            # K-Means로 19개 중심점(랜드마크) 찾기
            centers = extract_landmarks_kmeans(mask_2d, K=19)
            if centers is not None:
                landmark_list.append(centers)
                label_list.append(labels[i].item())
            else:
                print("활성화 픽셀이 19개 미만: 스킵합니다.")

all_dots = np.array(landmark_list, dtype=float)  # (N,19,2)
label_array = np.array(label_list)               # (N,)


# 5. 랜드마크 정렬 함수
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
    
    # 현재 처리 중인 인덱스
    current_idx = 0
    
    while current_idx < len(sorted_by_x):
        # 현재 x값
        current_x = sorted_by_x[current_idx, 1]
        
        # x값이 현재값과 10 이하 차이나는 점들의 인덱스 찾기
        similar_x_indices = []
        for i in range(current_idx, len(sorted_by_x)):
            if sorted_by_x[i, 1] - current_x <= 10:  # 20에서 10으로 수정
                similar_x_indices.append(i)
            else:
                break
        
        # 해당 점들을 y값으로 정렬
        if len(similar_x_indices) > 1:
            points_to_sort = sorted_by_x[similar_x_indices]
            sorted_by_y = points_to_sort[np.argsort(points_to_sort[:, 0])]
            result[current_idx:current_idx+len(similar_x_indices)] = sorted_by_y
        else:
            result[current_idx] = sorted_by_x[current_idx]
        
        current_idx += len(similar_x_indices)
    
    return result


# 6. 가장 가까운 점과 각도 계산 함수
def calculate_angles_between_nearest_points(dots):
    """
    dots: (19, 2) (y, x) 좌표
    각 점마다 '가장 가까운' 다른 점을 찾고,
    두 점을 잇는 선과 x축 사이의 각도(도 단위) 계산.
    """
    n_points = dots.shape[0]
    nearest_pts = []
    angles = []
    
    for i in range(n_points):
        current = dots[i]
        distances = np.sqrt(np.sum((dots - current)**2, axis=1))
        distances[i] = np.inf
        nearest_idx = np.argmin(distances)
        nearest_pt = dots[nearest_idx]
        
        dy, dx = nearest_pt - current
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        nearest_pts.append(nearest_pt)
        angles.append(angle_deg)
    
    return np.array(nearest_pts), np.array(angles)


# 7. 기하학적 특성 계산 함수
def calculate_geometric_features(dots):
    """
    추가적인 기하학적 특성 계산
    """
    features = []
    
    # 1. Centroid size
    centroid_point = np.mean(dots, axis=0)
    distances = np.sqrt(np.sum((dots - centroid_point)**2, axis=1))
    centroid_size = np.mean(distances)
    features.append(centroid_size)
    
    # 2. Convex Hull 면적
    hull = ConvexHull(dots)
    features.append(hull.area)
    
    # 3. 점들 간의 최대/최소 거리
    distances = []
    for i in range(len(dots)):
        for j in range(i+1, len(dots)):
            dist = np.sqrt(np.sum((dots[i] - dots[j])**2))
            distances.append(dist)
    features.extend([np.max(distances), np.min(distances)])
    
    # 4. 점들의 분포 특성
    features.extend([
        np.std(dots[:, 0]),  # y축 표준편차
        np.std(dots[:, 1]),  # x축 표준편차
        np.var(dots[:, 0]),  # y축 분산
        np.var(dots[:, 1])   # x축 분산
    ])
    
    return np.array(features)


# 8. 데이터 증강 함수
def augment_landmarks(dots, noise_level=0.05):
    """
    랜드마크 좌표에 노이즈를 추가하여 데이터 증강
    """
    noise = np.random.normal(0, noise_level, dots.shape)
    return dots + noise


# 9. 각도 -> sin/cos 변환
def convert_angles_to_sin_cos(angles_deg):
    """
    angles_deg: shape=(19,)  # degree
    각도 1개에 대해 [sin, cos] 2개로 확장 -> 최종 38차원
    """
    sin_cos_list = []
    for angle in angles_deg:
        rad = np.deg2rad(angle)
        sin_cos_list.append(np.sin(rad))
        sin_cos_list.append(np.cos(rad))
    return sin_cos_list


# 10. 피처 생성 및 데이터 증강
angle_features_sin_cos = []
augmented_features = []
augmented_labels = []

for i in range(len(all_dots)):
    # 원본 데이터 처리
    sorted_landmarks = sort_dots(all_dots[i])
    _, angles = calculate_angles_between_nearest_points(sorted_landmarks)
    
    # 기본 피처 (각도 + sin/cos)
    feature_38 = np.array(convert_angles_to_sin_cos(angles))
    
    # 기하학적 특성 추가
    geometric_features = calculate_geometric_features(sorted_landmarks)
    
    # 모든 피처 결합
    feature_all = np.concatenate([feature_38, geometric_features])
    angle_features_sin_cos.append(feature_all)
    
    # 데이터 증강 (3배)
    for _ in range(3):
        aug_landmarks = augment_landmarks(sorted_landmarks)
        _, aug_angles = calculate_angles_between_nearest_points(aug_landmarks)
        aug_feature_38 = np.array(convert_angles_to_sin_cos(aug_angles))
        aug_geometric_features = calculate_geometric_features(aug_landmarks)
        aug_feature_all = np.concatenate([aug_feature_38, aug_geometric_features])
        
        augmented_features.append(aug_feature_all)
        augmented_labels.append(label_array[i])

# 원본 데이터와 증강 데이터 합치기
X_all = np.vstack([np.array(angle_features_sin_cos), np.array(augmented_features)])
y_all = np.concatenate([label_array, np.array(augmented_labels)])

# 11. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 12. 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 13. 피처 선택
selector = SelectKBest(score_func=f_classif, k=30)  # 상위 30개 피처 선택
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# 14. 앙상블 모델 정의
svm = SVC(kernel='rbf', probability=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('rf', rf),
        ('knn', knn)
    ],
    voting='soft'
)

# 15. Grid Search for ensemble
param_grid = {
    'svm__C': [0.1, 1, 10, 30, 100],
    'svm__gamma': [0.01, 0.1, 0.3, 1, 10],
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'knn__n_neighbors': [3, 5, 7]
}

grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_

# 16. Stratified K-Fold 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train)):
    X_fold_train = X_train_selected[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_selected[val_idx]
    y_fold_val = y_train[val_idx]
    
    best_model.fit(X_fold_train, y_fold_train)
    fold_score = best_model.score(X_fold_val, y_fold_val)
    cv_scores.append(fold_score)
    
    print(f"Fold {fold+1} Score: {fold_score:.4f}")

print(f"Average CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# 17. 최종 테스트 세트 평가
y_pred = best_model.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)
print(f"[앙상블 분류] 최종 테스트 정확도: {acc*100:.2f}%")

# 18. 시각화 함수
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
    if centers1 is not None:
        plt.scatter(centers1[:,1], centers1[:,0], s=30, marker='x', c='red', label=f'Sample {sample_idx1}')
        for idx, (y, x) in enumerate(centers1):
            plt.annotate(str(idx), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')
    
    # 두 번째 이미지 (파란색 계열)
    plt.imshow(img2_np, cmap='gray', alpha=alpha2)
    if centers2 is not None:
        plt.scatter(centers2[:,1], centers2[:,0], s=30, marker='x', c='blue', label=f'Sample {sample_idx2}')
        for idx, (y, x) in enumerate(centers2):
            plt.annotate(str(idx), (x, y), xytext=(-5, -5), textcoords='offset points', 
                        fontsize=8, color='blue')
    
    plt.title(f"Overlapped Samples\nRed: Sample {sample_idx1} (Label={label1})\nBlue: Sample {sample_idx2} (Label={label2})")
    plt.legend()
    plt.show()

# 시각화 예시
visualize_overlapped_landmarks(dataset_test, all_dots, sample_idx1=0, sample_idx2=80, alpha1=0.7, alpha2=0.3) 