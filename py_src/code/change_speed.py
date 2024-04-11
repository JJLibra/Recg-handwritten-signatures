import librosa
import soundfile as sf
import numpy as np
import os

def change_speed(input_folder, output_folder, speed_factor=0.5):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有.wav文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            # 读取音频
            y, sr = librosa.load(file_path, sr=None)
            # 改变音频速度 - 注意，这种方法会改变音调
            y_changed = np.interp(np.arange(0, len(y), speed_factor), np.arange(0, len(y)), y)
            # 构造输出文件路径
            output_file_path = os.path.join(output_folder, file_name)
            # 保存处理后的音频文件
            sf.write(output_file_path, y_changed, sr)

# 示例用法
input_folder = '1data/111/wusiyuan'  # 替换为你的输入文件夹路径
output_folder = '1data/111/speed'  # 替换为你想要保存处理后音频的文件夹路径
change_speed(input_folder, output_folder)

from pydub import AudioSegment
import os

def change_speed_with_pydub(input_folder, output_folder, speed_factor=0.5):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有.wav文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            # 加载音频
            audio = AudioSegment.from_wav(file_path)
            # 改变音频速度
            playback_speed = 1 / speed_factor
            audio_with_changed_speed = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * playback_speed)
            }).set_frame_rate(audio.frame_rate)
            # 构造输出文件路径
            output_file_path = os.path.join(output_folder, file_name)
            # 保存处理后的音频文件
            audio_with_changed_speed.export(output_file_path, format="wav")

input_folder = '1data/111/wusiyuan'  # 替换为你的输入文件夹路径
output_folder = '1data/111/speed'  # 替换为你想要保存处理后音频的文件夹路径
change_speed_with_pydub(input_folder, output_folder)


