import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# 用于提取音频特征的函数
def extract_features(file_name):
    # 加载音频文件
    y, sr = librosa.load(file_name, sr=None)
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 额外特征
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y)) # 零交叉率
    spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) # 频谱质心
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)) # 频谱对比度
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) # 频谱滚降点
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y)) # 频谱平坦度
    
    # 结合所有特征
    combined_features = np.hstack((mfcc_mean, zero_crossing_rate, spectral_centroids, spectral_contrast, spectral_rolloff, spectral_flatness))
    # combined_features = np.hstack((mfcc_mean, zero_crossing_rate, spectral_centroids, spectral_contrast))
    
    return combined_features

# 用于存储特征和标签的列表
features = []
labels = []

# 初始化存储wav文件路径和标签的列表
path_and_labels = []

for label in range(1, 5):  # 标签1到4
    folder_path = f'data/{label}'
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
    
    for file in audio_files:
        # 添加路径和标签
        path_and_labels.append([file, label])

# 转换为DataFrame
df_path_and_labels = pd.DataFrame(path_and_labels, columns=['path', 'label'])
# 指定保存路径和文件名
csv_file_path_for_path_and_label = 'path_and_label.csv'
# 保存到CSV文件，不包含索引和列名
df_path_and_labels.to_csv(csv_file_path_for_path_and_label, index=False, header=False)
# 输出保存成功的消息
print(f"文件已保存到 {csv_file_path_for_path_and_label}")

# 对每个文件夹进行迭代
for label in range(1, 5):  # 标签1到4
    folder_path = f'data/{label}'
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
    
    for file in audio_files:
        # 提取特征
        extracted_features = extract_features(file)
        # 添加特征和标签
        features.append(extracted_features)
        labels.append(label)

# 转换为适合模型训练的格式
features = np.array(features)
labels = np.array(labels)

# 假定features和labels变量已经按照您之前的代码准备好
# 将特征和标签合并为DataFrame
df = pd.DataFrame(features)
df['label'] = labels  # 在DataFrame中添加标签列

# 保存到CSV文件
csv_file_path = 'feature_and_label.csv'  # 指定保存路径和文件名
df.to_csv(csv_file_path, index=False, header=False)  # 保存到CSV，不包含索引和列名

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化和训练模型
# model = SVC(kernel='linear')
# 使用找到的最佳参数初始化和训练模型
model = SVC(C=11.963459175608655, kernel='rbf', gamma=0.054344375598616296)
model.fit(X_train, y_train)

# 在测试集上评估模型
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# 在测试集上评估模型
predictions = model.predict(X_test)
# 输出准确度
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
# 输出精确度
precision = precision_score(y_test, predictions, average='weighted')
print("Precision:", precision)
# 输出召回率
recall = recall_score(y_test, predictions, average='weighted')
print("Recall:", recall)
# 输出F1得分
f1 = f1_score(y_test, predictions, average='weighted')
print("F1 Score:", f1)