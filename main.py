import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt

wav_path = './ljj_test.wav'
# 读取.wav文件
rate, data = wav.read(wav_path)

# 设计巴特沃斯滤波器参数
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 应用滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设置所需的截止频率
lowcut = 600.0
highcut = 1000.0

# 应用巴特沃斯带通滤波器
filtered_data = butter_bandpass_filter(data, lowcut, highcut, rate, order=6)

# 如果需要，保存处理后的数据到新的.wav文件
wav.write('ljj_handle6.wav', rate, filtered_data.astype(np.int16))
