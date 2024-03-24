import librosa.display
import numpy as np
import matplotlib.pyplot as plt

wav_path = './ljj_test.wav'

# 加载音频文件
y, sr = librosa.load(wav_path, sr=None)  # sr=None 保持原始采样率

# 计算梅尔频谱
# S = librosa.feature.melspectrogram(y, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.power_to_db(S, ref=np.max)

# 绘制梅尔频谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
