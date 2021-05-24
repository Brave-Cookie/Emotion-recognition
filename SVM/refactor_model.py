import os
import librosa
import joblib

# 내 데이터셋의 절대경로 지정
file_path = 'C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/dataset/cut4/'
file_names = os.listdir(file_path)
files = []
for file_name in file_names:
    files.append(file_path+file_name)

print(' -------------------------------------------- Dataset 경로 저장 완료 -------------------------------------------- ')

dataset=[]
label=[] 

for file in files:
    signal, sr = librosa.load(file, sr=16000)
    mfcc= librosa.feature.mfcc(signal, sr, n_fft=400, hop_length=160, n_mfcc=36)
    dataset.append(mfcc.reshape(-1)) # 1차원 배열로 반환
    label.append((os.path.basename(file)).split('_')[1]) # 파일의 이름에서 감정을 나타내는 부분을 추출해서 저장


print(' -------------------------------------------- 전처리 완료 -------------------------------------------- ')

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(label)
y = encoder.transform(label)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 학습 데이터와 훈련 데이터 split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 42, shuffle = True)

# 커널 설정
clf = SVC(C=10, kernel = 'rbf', probability=True)

# MinMax Scaler를 통해 특징 벡터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(' -------------------------------------------- scaler 학습 완료 -------------------------------------------- ')

# 조정된 데이터로 SVM 학습
clf.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print(clf.score(X_test_scaled, y_test))

print(' -------------------------------------------- 모델 학습 완료 -------------------------------------------- ')

# 학습한 모델 파일로 저장
joblib.dump(clf, 'model2.pkl')

print(' -------------------------------------------- pkl 저장 완료 -------------------------------------------- ')

# 파일로 저장된 모델 불러와서 예측
clf_from_joblib = joblib.load('model.pkl') 
clf_from_joblib.predict(X_test_scaled)
