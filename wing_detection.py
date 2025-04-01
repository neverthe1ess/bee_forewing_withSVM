from ultralytics import YOLO

def load_model(model_path='path/to/your/yolov11.pt'):
    """
    YOLO 모델을 불러옵니다.
    
    model_path: 학습된 YOLOv11 가중치 파일 경로 (커스텀 모델 사용 시)
    """
    # 아래 코드는 YOLOv5 형식의 torch.hub 로드 예제입니다.
    # 사용 중인 YOLOv11 모델에 맞게 로드 방법을 수정하세요.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

def train_model():
    model = YOLO("yolo11m.pt")
    
    results = model.train(data="coco.yaml", epochs = 100, imgsz=640)

def detect_and_crop_wing(image_path, model, class_id=0, crop_size=(224, 224), conf_threshold=0.5):
    """
    이미지에서 날개(wing) 객체를 탐지한 후, 해당 영역을 크롭하고 전처리합니다.
    
    Args:
        image_path (str): 입력 이미지 경로.
        model: 불러온 YOLO 모델.
        class_id (int): 모델에서 "wing" 객체에 해당하는 클래스 인덱스 (예제에서는 0).
        crop_size (tuple): 크롭 후 리사이즈할 이미지 크기 (width, height).
        conf_threshold (float): 탐지 신뢰도 임계값.
    
    Returns:
        전처리된 날개 영역 이미지 리스트 (numpy 배열).
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 찾을 수 없거나 읽기에 실패했습니다.")
    
    # 모델 추론 수행
    results = model(img)
    # results.xyxy[0]에는 [x1, y1, x2, y2, confidence, class] 형태의 탐지 결과가 있습니다.
    cropped_wings = []
    for *box, conf, cls in results.xyxy[0]:
        if conf < conf_threshold:
            continue  # 신뢰도 낮은 탐지 제외
        if int(cls) == class_id:
            # 바운딩 박스 좌표 정수형 변환
            x1, y1, x2, y2 = map(int, box)
            # 날개 영역 크롭
            crop = img[y1:y2, x1:x2]
            # 전처리: 지정된 크기로 리사이즈
            crop_resized = cv2.resize(crop, crop_size)
            # 추가 전처리 (예: 정규화, 색상 변환 등) 필요 시 이곳에 구현
            cropped_wings.append(crop_resized)
    
    return cropped_wings

if __name__ == "__main__":
    # 모델 불러오기 (가중치 파일 경로를 실제 경로로 변경)
    model = load_model("./saved_models/best.pt")
    
    # 탐지할 이미지 경로 (실제 이미지 파일 경로로 변경)
    image_path = "path_to_your_image.jpg"
    
    # 날개 객체 탐지 및 크롭 수행
    wing_crops = detect_and_crop_wing(image_path, model)
    
    # 크롭된 날개 이미지 저장 또는 후처리
    for idx, crop in enumerate(wing_crops):
        output_path = f"cropped_wing_{idx}.jpg"
        cv2.imwrite(output_path, crop)
        print(f"저장 완료: {output_path}")