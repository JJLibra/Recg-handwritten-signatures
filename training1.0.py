# 贝叶斯
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# 从feature_vectors.csv中导入数据
data = pd.read_csv('features/feature.csv')
# data = pd.read_csv('p4_test_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = GaussianNB()
# model = MultinomialNB()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')  # 使用加权平均
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 决策树
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# 从feature_vectors.csv中导入数据
# data = pd.read_csv('features/feature.csv')
data = pd.read_csv('p4_test_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')  # 使用加权平均
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 随机森林
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# 从feature_vectors.csv中导入数据
# data = pd.read_csv('features/feature_NB.csv')
data = pd.read_csv('p4_test_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 建立随机森林模型
rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # 可根据需要调整参数
# 训练模型
rfc.fit(X_train, y_train)
# 预测测试集标签
y_pred = rfc.predict(X_test)

# 计算精确度、召回率和 F1 分数
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 梯度提升分类器
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

# 从feature_vectors.csv中导入数据
data = pd.read_csv('features/feature.csv')
# data = pd.read_csv('p4_test_vectors.csv')
# data = pd.read_csv('feature_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 创建梯度提升分类器
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
# 在训练集上拟合模型
gbc.fit(X_train, y_train)
# 预测测试集标签
y_pred = gbc.predict(X_test)

# 计算精确度、召回率和 F1 分数
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# XGBoost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 从feature_vectors.csv中导入数据
data = pd.read_csv('features/feature.csv')
# data = pd.read_csv('p4_test_vectors.csv')
# data = pd.read_csv('feature_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 创建 LabelEncoder 对象
le = LabelEncoder()

# 将字符串标签编码为整数
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 创建并训练 XGBoost 模型
model = xgb.XGBClassifier()
model.fit(X_train, y_train_encoded)

# 预测测试集标签
y_pred_encoded = model.predict(X_test)

# 计算精确度、召回率和 F1 分数
precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted')
f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

data = pd.read_csv('features/feature.csv')
# data = pd.read_csv('p4_test_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 创建模型
model = lgb.LGBMClassifier(verbose=-1)
# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
# accuracy = (y_pred == y_test).mean()
# print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 支持向量机
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# 从feature_vectors.csv中导入数据
# data = pd.read_csv('features/feature.csv')
data = pd.read_csv('p4_test_vectors.csv')
# data = pd.read_csv('feature_vectors.csv')

# 划分训练集和测试集
X = data.drop(columns=['label'])  # 特征向量
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 使用支持向量机（SVM）模型
# model = SVC(kernel='linear')
model = SVC(kernel='linear', C=0.001)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')  # 使用加权平均
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 超参数优化
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, randint

# 设置SVC模型
model = make_pipeline(StandardScaler(), SVC())

# 定义要搜索的参数分布
param_distributions = {
    'svc__C': expon(scale=10),  # 对于连续参数，可以使用分布
    'svc__kernel': ['linear', 'rbf'],  # 对于分类参数，仍然是列表
    'svc__gamma': expon(scale=.1)  # 使用分布来随机采样
}

# 创建RandomizedSearchCV对象
random_search = RandomizedSearchCV(model, param_distributions, n_iter=100, cv=5, scoring='accuracy')

# 进行搜索
random_search.fit(X_train, y_train)

# 打印最佳参数
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# 学习曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 定义计算学习曲线的函数
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 定义模型
# model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model = make_pipeline(StandardScaler(), SVC(C=11.963459175608655, kernel='rbf', gamma=0.054344375598616296))

# 绘制学习曲线
plot_learning_curve(model, "Learning Curve (SVC, linear kernel)", X_train, y_train, cv=5, n_jobs=-1)

plt.show()



