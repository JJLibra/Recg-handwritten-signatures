import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

wav_path = './ljj_handle4.wav'

# 读取.wav文件
samplerate, data = wavfile.read(wav_path)

# fft_res = np.fft.fft(samplerate)  # 傅里叶变换
# fft_res = abs(fft_res)[:len(fft_res)//2] / len(samplerate) * 2

# plt.figure(figsize=(8,4))
# # 原始信号
# plt.subplot(2,1,1)
# plt.subplots_adjust(wspace=0,hspace=0.5) # 用来调整间距的
# plt.plot(samplerate,linewidth=0.5)
# plt.title("时域信号")

# # 频谱图
# plt.subplot(2,1,2)
# plt.plot(fft_res,linewidth=0.5)
# plt.title("FFT")


# 计算时间轴
times = np.arange(len(data)) / float(samplerate)

# 绘制波形图
plt.figure(figsize=(15, 5))
plt.fill_between(times, data, color='k')
plt.xlim(times[0], times[-1])
plt.xlabel('Time (s)')
plt.ylabel('A')
plt.title('r')
plt.show()
