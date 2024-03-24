import os
import csv
import librosa
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 标记音频文件并将路径和标签存储在CSV文件中
folders = ['cc', 'lijunjie', 'wusiyuan', 'xuzhaoqi']  # 文件夹名称，按首字母排的
labels = [1, 2, 3, 4]  # 对应的标签

# 创建CSV文件以存储音频文件路径和标签
with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['audio_path', 'label'])
    
    # 遍历每个文件夹并标记音频文件
    for folder, label in zip(folders, labels):
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                audio_path = os.path.join(folder, filename)
                writer.writerow([audio_path, label])

# 从音频文件中提取MFCC特征
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs.flatten()  # 将MFCC特征展平

# 读取数据集CSV
df = pd.read_csv('dataset.csv')
df['mfcc'] = df['audio_path'].apply(extract_features)  # 应用特征提取

# 准备分类器的数据
# 创建一个新的DataFrame，包含展平的特征
features_df = pd.DataFrame(df['mfcc'].tolist())
features_df['label'] = df['label']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_df.drop('label', axis=1), features_df['label'], test_size=0.2, random_state=42)

# 初始化朴素贝叶斯分类器
gnb = GaussianNB()
# 训练模型
gnb.fit(X_train, y_train)

# 进行预测
y_pred = gnb.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
# 打印准确率
print(f"朴素贝叶斯分类器的准确率为: {accuracy}")




# 最后检测签名人身份的代码
def predict_signature_identity(audio_path, model, feature_extractor):
    # 提取音频文件的特征
    features = feature_extractor(audio_path)
    
    # 将特征转换为模型可以接受的格式（2D数组）
    features = features.reshape(1, -1)
    
    # 使用模型进行预测
    prediction = model.predict(features)
    
    # 返回预测的标签
    return prediction[0]

# 假设我们有一个新的手签音频文件 "new_signature.wav"
new_audio_path = "new_signature.wav"

# 使用预测函数来检测签名人身份
predicted_label = predict_signature_identity(new_audio_path, gnb, extract_features)

# 打印预测结果
print(f"预测的手签音频 {new_audio_path} 的签名人身份标签为: {predicted_label}")
