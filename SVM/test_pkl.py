import joblib
import librosa


# 데이터셋 있는곳 절대경로
file_path = 'C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/dataset/cut4/'
# 테스트해볼 데이터셋 파일명
file_names = 'cuts4_fear_M_f15_1.wav'
file = file_path + file_names

signal, sr = librosa.load(file, sr=16000)
mfcc = librosa.feature.mfcc(signal, sr, n_fft=400, hop_length=160, n_mfcc=36)
mfcc = mfcc.reshape(1, -1)

# pkl 파일 load
clf_from_joblib = joblib.load('C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/SVM/model.pkl')
result = clf_from_joblib.predict(mfcc)

print(result)