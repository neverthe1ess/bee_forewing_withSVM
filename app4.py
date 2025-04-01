import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1) 텍스트 파일 읽기
#    - sep='\s+' : 공백(whitespace) 구분자를 의미
#    - header=0  : 첫 번째 줄을 컬럼명으로 사용
df = pd.read_csv("scores.txt", sep='\s+', header=0)

# 2) 혹시 컬럼명이 정확히 지정되지 않았다면, 원하는 이름으로 재설정 가능
#    (데이터 첫 줄의 스펠링이 'subspeices'인지 'subspecies'인지 확인 후 맞춰 주세요)
df.columns = ["Id", "subspeices", "CV1", "CV2", "CV3", "CV4", "CV5"]

# 3) DataFrame 확인
print(df.head())   # 앞부분 5행
print(df.tail())   # 뒷부분 5행


# 2) CSV(또는 Excel)로부터 데이터 불러오기 (예: cva_scores.csv)
# df = pd.read_csv("cva_scores.csv")  # 실제 파일 경로로 대체

# 여기서는 예시용으로 랜덤 데이터를 생성해 보겠습니다.
np.random.seed(42)
num_samples = 100

# 3) 독립변수(X)와 종속변수(y) 분리
X = df[['CV1', 'CV2', 'CV3', 'CV4', 'CV5']].values  # CV1~CV5만 사용
y = df['subspeices'].values

# 4) 데이터 분할(학습:테스트) - 예: 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=11,
                                                    stratify=y)

# 5) 분류기 선언
lda = LinearDiscriminantAnalysis()
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# 6) 교차검증(Cross-validation)으로 분류 정확도 확인 (K=5 예시)
lda_scores = cross_val_score(lda, X_train, y_train, cv=5)
rf_scores  = cross_val_score(rf,  X_train, y_train, cv=5)
svm_scores = cross_val_score(svm, X_train, y_train, cv=5)

print("LDA CV accuracy: %.3f ± %.3f" % (lda_scores.mean(), lda_scores.std()))
print("RF  CV accuracy: %.3f ± %.3f" % (rf_scores.mean(),  rf_scores.std()))
print("SVM CV accuracy: %.3f ± %.3f" % (svm_scores.mean(), svm_scores.std()))

# 7) 최종 모델 훈련 후 테스트 세트 성능 확인 (예: RF)
rf.fit(X_train, y_train)
test_accuracy = rf.score(X_test, y_test)
print("Random Forest test accuracy:", round(test_accuracy, 3))

from sklearn.metrics import confusion_matrix, classification_report

y_pred = rf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))