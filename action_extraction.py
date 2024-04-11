# 音频基础信息提取
import os
import librosa

# 路径到你的音频文件夹
folder_path = 'data/1'

# 读取文件夹中所有音频文件的路径
audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

# 使用Librosa读取音频文件的基本信息
for file in audio_files:
    y, sr = librosa.load(file, sr=None)  # 加载音频文件，sr=None表示保留原始采样率
    duration = librosa.get_duration(y=y, sr=sr)  # 获取音频持续时间
    print(f"File: {os.path.basename(file)}, Sample Rate: {sr}, Duration: {duration:.2f} seconds")


import librosa
from scipy.io import wavfile

# 加载WAV文件
audio_path = 'test/xuzhaoqi_ljj.wav'
y, sr = librosa.load(audio_path)

window_size = 10 * sr  # 每10秒的采样数
segments = [y[i:i+window_size] for i in range(0, len(y), window_size)]

# 假设文件名为 example_0.wav, example_1.wav, ...
for i, segment in enumerate(segments):
    output_path = f'test/xuzhaoqi_ljj/xuzhaoqi_{i}.wav'

    # 将每个部分保存为 WAV 文件
    wavfile.write(output_path, sr, segment)

