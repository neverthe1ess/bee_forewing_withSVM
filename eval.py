import torch
import torchvision.transforms as transforms

from app import procrustes_transform 
import joblib
from model import UNet
from util import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
svm_model = joblib.load("SVM.joblib")
net = UNet().to(device)
batch_size = 4

input_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

dataset_test = Dataset(data_dir='./dataset',
                       transform=input_transforms)
loader_test = DataLoader(dataset_test, batch_size=batch_size,
                         shuffle=False, num_workers=0)

def predict_species(output):
    """
    UNet output을 받아 SVM을 통해 꿀벌 아종을 예측하는 함수.
    """
    landmarks = extract_landmarks(output)

    if landmarks is None:
        print("랜드마크를 충분히 감지하지 못했습니다.")
        return

    # Procrustes 정규화 적용
    landmarks = procrustes_transform(landmarks.reshape(1, -1))

    # SVM 예측 수행
    prediction = svm_model.predict(landmarks)
    probabilities = svm_model.predict_proba(landmarks)

    print(f"🐝 예측된 아종: {prediction[0]}")
    print("📊 예측 확률:")
    for species, prob in zip(svm_model.classes_, probabilities[0]):
        print(f" - {species}: {prob:.2%}")

# 예제: UNet 추론 결과를 받아 SVM 분류 수행
with torch.no_grad():
    net.eval()
    for batch, data in enumerate(loader_test, 1):
        input = data['input'].to(device)
        output = net(input)  # UNet 출력 (segmentation map)

        # UNet 출력 기반으로 SVM 예측 실행
        predict_species(output)