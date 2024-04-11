# 预测成功次数和总次数的初始化
success_counts = {1: 0, 2: 0, 3: 0, 4: 0}
total_counts = {1: 0, 2: 0, 3: 0, 4: 0}

# 假设目标文件夹路径为 'data/to_predict'
# target_folder_path = '1data/111/chence'
# target_folder_path = '1data/111/wusiyuan'
# target_folder_path = '1data/111/xuzhaoqi'
# target_folder_path = '1data/111/speed/0.5'
# target_folder_path = '1data/111/speed/2'
target_folder_path = '1data/111/speed/3'

# 读取目标文件夹中所有WAV音频文件的路径
target_audio_files = [os.path.join(target_folder_path, file) for file in os.listdir(target_folder_path) if file.endswith('.wav')]

# 根据文件名首字母确定正确的标签
def get_correct_label(filename):
    if filename.startswith('c'):
        return 1
    elif filename.startswith('l'):
        return 2
    elif filename.startswith('w'):
        return 3
    elif filename.startswith('x'):
        return 4
    else:
        return None  # 如果不符合任何条件，返回None
        
incorrect_predictions = []

# 遍历每个文件，进行预测并更新成功次数和总次数
for file in target_audio_files:
    correct_label = get_correct_label(os.path.basename(file))
    if correct_label is None:  # 如果无法确定正确的标签，跳过此次循环
        continue

    # 提取特征并进行预测（保持之前的代码不变）
    extracted_features = extract_features(file).reshape(1, -1)
    extracted_features = scaler.transform(extracted_features)
    prediction = model.predict(extracted_features)[0]

    # 更新总次数
    total_counts[correct_label] += 1

    # 如果预测正确，更新成功次数
    if prediction == correct_label:
        success_counts[correct_label] += 1
    else:
        # 如果预测错误，将文件名添加到incorrect_predictions列表中
        incorrect_predictions.append(os.path.basename(file))

# 计算并打印每个分类的预测成功率及平均成功率
success_rates = []

for label in range(1, 5):
    if total_counts[label] > 0:
        success_rate = success_counts[label] / total_counts[label] * 100
        success_rates.append(success_rate)
        print(f"Label {label} Prediction Success Rate: {success_rate:.2f}%")
    else:
        print(f"Label {label}: No files to predict.")

# 计算四个类别的平均成功率
if success_rates:
    average_success_rate = sum(success_rates) / len(success_rates)
    print(f"\nAverage Success Rate across all classes: {average_success_rate:.2f}%")
else:
    print("No predictions were made for any class.")

if incorrect_predictions:
    print("Incorrectly predicted files:")
    for file in incorrect_predictions:
        print(file)
else:
    print("No incorrect predictions.")