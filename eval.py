import torch

# 저장된 모델 로드
from svm import procrustes_transform, extract_landmarks
import joblib

svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

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

    # 스케일링 적용
    landmarks_scaled = scaler.transform(landmarks)

    # SVM 예측 수행
    prediction = svm_model.predict(landmarks_scaled)
    probabilities = svm_model.predict_proba(landmarks_scaled)

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