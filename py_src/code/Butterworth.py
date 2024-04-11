import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, lfilter
# 读入音频文件
# samplerate, data = wavfile.read('first_collection/240330/chence2.wav')
samplerate, data = wavfile.read('test/place1/xzq.wav')

# 巴特沃斯滤波函数
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 参数
order = 6
cutoff = 1200  # 截止频率
filtered_data = butter_lowpass_filter(data, cutoff, samplerate, order)
wavfile.write('test/place1/filtered_xzq.wav', samplerate, filtered_data.astype(np.int16))




