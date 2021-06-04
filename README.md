# 🎯 Emotion-recognition

음성 데이터를 4초 길이로 자른 후, **MFCC**를 사용해 특징 벡터를 추출합니다. 추출한 특징 벡터는 감정 별로 값의 범위가 다양하기 때문에
Scaler를 사용해 표준화시킵니다. Scaler는 **MinMax Scaler**를 사용했습니다. 이후 **SVM 모델을 학습**시킵니다. 

완성된 모델의 정확도는 약 **85%이며**, 각 감정에 대한 예측 결과는 아래와 같습니다.


![true](https://user-images.githubusercontent.com/48341341/116882185-1c346b00-ac5f-11eb-8eb1-82d350d2c71a.PNG)
