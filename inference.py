import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix, accuracy_score
from model import UNet
from util import load
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import seaborn as sns


# 1. 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# U-Net 모델 로드
ckpt_dir = './checkpoint'
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
net.eval()

# SVM 모델 및 스케일러 로드
model_dir = './saved_models'
svm_model = joblib.load(os.path.join(model_dir, 'svm_model.joblib'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

# 2. 이미지 전처리 설정
input_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# 3. 랜드마크 추출 함수
def extract_landmarks_kmeans(binary_mask, K=19):
    """
    binary_mask: (H, W) 0/1 이진 마스크 (numpy)
    K: 찾고자 하는 군집(랜드마크) 개수
    """
    coords = np.argwhere(binary_mask)  # shape=(N, 2)
    num_pixels = coords.shape[0]
    if num_pixels < K:
        return None
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)  # (K, 2) 정수 좌표 (y, x)
    return centers

# 4. 랜드마크 정렬 함수
def sort_dots(dots):
    """
    dots: (19, 2) 형태의 랜드마크 좌표 (y, x)
    x좌표 기준 오름차순 정렬
    x좌표 픽셀 차이가 10 이하인 경우 y좌표로 정렬
    """
    sorted_by_x = dots[np.argsort(dots[:, 1])]
    result = np.zeros_like(sorted_by_x)
    current_idx = 0
    
    while current_idx < len(sorted_by_x):
        current_x = sorted_by_x[current_idx, 1]
        similar_x_indices = []
        
        for i in range(current_idx, len(sorted_by_x)):
            if sorted_by_x[i, 1] - current_x <= 10:
                similar_x_indices.append(i)
            else:
                break
        
        if len(similar_x_indices) > 1:
            points_to_sort = sorted_by_x[similar_x_indices]
            sorted_by_y = points_to_sort[np.argsort(points_to_sort[:, 0])]
            result[current_idx:current_idx+len(similar_x_indices)] = sorted_by_y
        else:
            result[current_idx] = sorted_by_x[current_idx]
        
        current_idx += len(similar_x_indices)
    
    return result

# 5. 각도 계산 함수
def calculate_angles_between_nearest_points(dots):
    """
    dots: (19, 2) (y, x) 좌표
    각 점마다 '가장 가까운' 다른 점을 찾고,
    두 점을 잇는 선과 x축 사이의 각도(도 단위) 계산.
    """
    diff = dots[:, np.newaxis, :] - dots[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    
    nearest_indices = np.argmin(distances, axis=1)
    nearest_pts = dots[nearest_indices]
    
    dy = nearest_pts[:, 0] - dots[:, 0]
    dx = nearest_pts[:, 1] - dots[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    
    return nearest_pts, angles

# 6. 기하학적 특성 계산 함수
def calculate_geometric_features(dots):
    """
    최적화된 기하학적 특성 계산
    """
    features = []
    
    # 1. Centroid size
    centroid_point = np.mean(dots, axis=0)
    distances = np.sqrt(np.sum((dots - centroid_point)**2, axis=1))
    centroid_size = np.mean(distances)
    features.append(centroid_size)
      
    return np.array(features)

# 7. 각도 -> sin/cos 변환
def convert_angles_to_sin_cos(angles_deg):
    """
    angles_deg: shape=(19,)  # degree
    각도 1개에 대해 [sin, cos] 2개로 확장 -> 최종 38차원
    """
    rad = np.deg2rad(angles_deg)
    return np.column_stack([np.sin(rad), np.cos(rad)]).flatten()

# 8. 시각화 함수
def visualize_landmarks(image, landmarks, prediction):
    """
    이미지와 랜드마크를 시각화하는 함수
    """
    plt.figure(figsize=(10, 8))
    
    # 이미지 준비
    if isinstance(image, torch.Tensor):
        img_np = image.numpy()
        if img_np.shape[0] == 1:
            img_np = img_np[0]
        else:
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        img_np = np.array(image)
    
    # 이미지 표시
    plt.imshow(img_np, cmap='gray')
    
    # 랜드마크 표시
    if landmarks is not None:
        plt.scatter(landmarks[:,1], landmarks[:,0], s=30, marker='x', c='red')
        for idx, (y, x) in enumerate(landmarks):
            plt.annotate(str(idx), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')
    
    # 제목 설정
    plt.title(f"Predicted: {prediction}")
    plt.show()

# 9. 추론 함수
def predict_species(image):
    """
    단일 이미지에 대한 랜드마크 예측 및 분류
    """
    # 이미지 전처리
    if isinstance(image, torch.Tensor):
        image = image.unsqueeze(0)  # 배치 차원 추가
    else:
        image = torch.from_numpy(image).unsqueeze(0)
    
    # U-Net 추론
    with torch.no_grad():
        output = net(image.to(device))
        binarized = (output > 0.5).float()
        mask_2d = binarized[0, 0].cpu().numpy()
    
    # 랜드마크 추출
    centers = extract_landmarks_kmeans(mask_2d, K=19)
    if centers is None:
        print("랜드마크를 충분히 감지하지 못했습니다.")
        return None, None, None
    
    # 피처 추출
    sorted_landmarks = sort_dots(centers)
    _, angles = calculate_angles_between_nearest_points(sorted_landmarks)
    
    # 기본 피처
    feature_38 = convert_angles_to_sin_cos(angles)
    
    # 기하학적 특성
    geometric_features = calculate_geometric_features(sorted_landmarks)
    
    # 모든 피처 결합
    feature_all = np.concatenate([feature_38, geometric_features])
    
    # 스케일링 및 예측
    feature_scaled = scaler.transform(feature_all.reshape(1, -1))
    prediction = svm_model.predict(feature_scaled)
    probabilities = svm_model.predict_proba(feature_scaled)
    
    return prediction[0], centers, probabilities[0]

# 10. 메인 추론 루프
def main():
    true_labels = []
    predicted_labels = []
    class_list = ["A", "D", "E", "F", "G", "H"]
    class_dict = {"A" : 0, "D" : 1, "E" : 2, "F" : 3, "G": 4, "H": 5}

    print("추론 시작...")
   
    # 테스트할 이미지 디렉토리
    data_dir = './data_test'
    
    # 디렉토리의 모든 이미지 파일 처리
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Error: {data_dir} 디렉토리에 이미지 파일이 없습니다.")
        return
    
    for image_file in image_files:
        # 이미지 로드
        image_path = os.path.join(data_dir, image_file)
        image = Image.open(image_path).convert('L')  # 흑백으로 변환
        
        # 실제 레이블 추출 (파일명에서 첫 번째 문자)
        true_label_name = image_file[0]  # 0-based index로 변환

        true_label = class_dict[true_label_name]

        # 이미지 전처리
        image_tensor = input_transforms(image)
        
        # 예측 수행
        prediction, landmarks, probabilities = predict_species(image_tensor)
        
        if prediction is not None:
            # 예측 결과 저장
            true_labels.append(true_label)
            predicted_labels.append(prediction - 1)  # 0-based index로 변환

            print(f"\n이미지: {image_file}")
            print(f"🐝 예측된 아종: {class_list[prediction - 1]}")
            print("📊 예측 확률:")
            for species, prob in zip(svm_model.classes_, probabilities):
                print(f" - {species}: {prob:.2%}")
            
            # 시각화
            visualize_landmarks(image_tensor, landmarks, prediction)
        else:
            print(f"\n이미지: {image_file}")
            print("❌ 랜드마크 추출 실패")
    
    print("\n추론 완료!")

    # 정확도 계산
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # 혼동 행렬 계산 및 시각화
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_list,
                yticklabels=class_list)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print(f"\n전체 정확도: {accuracy:.2%}")
if __name__ == "__main__":
    main() 
