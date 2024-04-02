import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import butter, lfilter

sound = AudioSegment.from_wav("test/filtered_lijunjie.wav")
# 签名开始和结束的时间，例如开始于5000毫秒，结束于10000毫秒
start_time = 5000
end_time = 10000
# 切割音频
cut_sound = sound[start_time:end_time]

cut_sound.export("test/lijunjie/18.wav", format="wav")

# 读取WAV文件
samplerate, data = wavfile.read('test/lijunjie/18.wav')
times = np.arange(len(data)) / float(samplerate)
# 绘制波形图
plt.figure(figsize=(15, 5))
plt.plot(times, data)
plt.title('WAV File Waveform')
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.xlim(0, times[-1])
plt.show()