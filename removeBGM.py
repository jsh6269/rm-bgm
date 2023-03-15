
# python을 사용. librosa, numpy, soundfile 모듈을 다운받아야 실행가능
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

fire = '여기에는 배경음악과 스킬보이스가 섞인 오디오 파일의 경로를 적는다'
fireA, sr = librosa.load(fire)

bgm = '여기에는 배경음악만 존재하는 오디오 파일의 경로를 적는다'
bgmA, sr = librosa.load(bgm)

# done이 False이면 파형의 그래프를 그려준다.
# done이 True이면 a1, a2의 값에 의존하여 싱크를 맞춘 상태에서 두 파형의 차를 계산하여 음악파일로 저장한다
done = True

# 파형의 특징적인 부분에 주목한다. 싱크가 정확히 일치하는 두 time 값을 a1, a2에 저장한다.
a1, a2 = 5571, 5635
l1, l2 = len(fireA), len(bgmA)
if done:
    # 두 음악파일의 길이를 동일하게 맞춘다.
    if l1-a1 > l2-a2:
        fireA, bgmA = fireA[:a1+l2-a2], bgmA[:a2+l2-a2]
    elif l1-a1 < l2-a2:
        fireA, bgmA = fireA[:a1+l1-a1], bgmA[:a2+l1-a1]
    if a1 < a2:
        bgmA = bgmA[a2-a1:]
    elif a1 > a2:
        fireA = fireA[a1-a2:]

print(fireA.shape, bgmA.shape)

timeA = list(range(len(fireA)))
timeB = list(range(len(bgmA)))

hop = int(10 / 1000 * sr)
win = int(25 / 1000 * sr)

# 그래프의 x축의 최대범위를 sized로 저장한다
sized = 20000
plt.subplot(2, 1, 1)
plt.plot(timeA[:sized], fireA[:sized])
plt.subplot(2, 1, 2)
plt.plot(timeB[:sized], bgmA[:sized])

if not done:
    plt.show()

fireB = librosa.stft(fireA, hop_length=hop, win_length=win)
fire_mag, fire_phase = np.abs(fireB), np.angle(fireB)

bgmB = librosa.stft(bgmA, hop_length=hop, win_length=win)
bgm_mag, bgm_phase = np.abs(bgmB), np.angle(bgmB)

# 3 대신 적당한 숫자로 매번 변경해가며 optimized point를 찾아야한다
bgm_mag = 3 * bgm_mag

new_mag = np.maximum(fire_mag - bgm_mag, 0)
new_out = new_mag * np.exp(fire_phase * 1j)
x_out = librosa.istft(new_out, hop_length=hop, win_length=win)

# sample.wav 파일로 결과가 저장된다.
sf.write('./sample.wav', x_out, sr)
