from pydub import AudioSegment,silence
import librosa

file_path = 'C:/Users/NB1/Desktop/PROGRAM/GitWorkSpace/CapstoneDesign_2021/Emotion-recognition/dataset/test_wav/'
file_name = '1831732294.wav'
file = file_path + file_name

# pydub로 묵음 구간 모두 추출
myaudio = AudioSegment.from_wav(file)
dBFS=myaudio.dBFS
silence = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=dBFS-16)
silence = [((start/1000),(stop/1000)) for start,stop in silence] #in sec
print(silence)


signal, sr = librosa.load(file, sr=16000)
print(signal)

# 묵음 구간을 없앤 실제 음성 구간 파싱
section_list = []
for start, end in silence:
    section_list.extend([start, end])
section_list = section_list[1:-1]

# 실제 음성 구간에서 2개씩 묶어준다.
new_section = []
for i, section in enumerate(section_list):
    if i%2 ==0:
        new_section.append([section, section_list[i+1]])
print(new_section)

# 실제 음성 구간을 추출, 모두 합쳐줌
reform_signal = []
for start, end in new_section:
    reform_signal.extend(signal[int(start*16000) : int(end*16000)])

audio_len = librosa.get_duration(reform_signal, sr)
print(audio_len)