import joblib
import librosa

# scaler pkl 파일 load 
scaler_from_joblib = joblib.load('C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/SVM/scaler.pkl')
# clf pkl 파일 load
clf_from_joblib = joblib.load('C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/SVM/model.pkl')

# 데이터셋 있는곳 절대경로
file_path = 'C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/dataset/cut4/'
# 테스트해볼 데이터셋 파일명
file_names = 'cuts1_anger_M_a16_4.wav'
file = file_path + file_names

# 음성파일 전처리 진행 
# load -> mfcc추출 -> scale 변환
signal, sr = librosa.load(file, sr=16000)
mfcc = librosa.feature.mfcc(signal, sr, n_fft=400, hop_length=160, n_mfcc=36)
mfcc = mfcc.reshape(-1)
mfcc = scaler_from_joblib.transform([mfcc])

# 결과는?
result = clf_from_joblib.predict(mfcc)
print(result[0])