import pandas as pd
from sklearn.model_selection import train_test_split  # 划分数据集与测试集
from sklearn import metrics  # 分类结果评价函数
from sklearn.metrics import accuracy_score  # 导入评分模块
from sklearn import svm  # 导入算法模块
from sklearn.ensemble import (
    RandomForestClassifier,
)  # 导入sklearn库的RandomForestClassifier函数
import argparse

# -------------------------------parse args
parser = argparse.ArgumentParser(description="wash goscanner results and get features")
parser.add_argument("--trueFeatures", "-t", help="features.csv from https", type=str)
parser.add_argument(
    "--falseFeatures", "-f", help="features.csv from other tls", type=str
)
parser.add_argument(
    "--tlsVersion",
    "-v",
    help="choose a version of tls(tls1.x) to test instead of all together",
    type=str,
)
parser.add_argument(
    "--withPort", "-p", help="whether with port as a feature", action="store_true"
)
args = parser.parse_args()

# --------------------------------gen dataset
tdata = pd.read_csv(args.trueFeatures)
fdata = pd.read_csv(args.falseFeatures)

# port as features?
if not args.withPort:
    tdata.drop(columns="port", inplace=True)
    fdata.drop(columns="port", inplace=True)

# use single tls version
protocol = 0
if args.tlsVersion == "tls1.0":
    protocol = 769
elif args.tlsVersion == "tls1.1":
    protocol = 770
elif args.tlsVersion == "tls1.2":
    protocol = 771
elif args.tlsVersion == "tls1.3":
    protocol = 772

if protocol != 0:
    tdata = tdata[tdata["protocol"] == protocol]
    fdata = fdata[fdata["protocol"] == protocol]
    tdata.drop(columns="protocol", inplace=True)
    fdata.drop(columns="protocol", inplace=True)


# add labels
tdata["is_https"] = 1
fdata["is_https"] = 0
# gen dataset
dataset = pd.concat([tdata, fdata])

# ---------------------training
train = dataset.drop(["is_https"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    train, dataset["is_https"], random_state=1, train_size=0.8
)

# svm_linear = svm.SVC(C=1, kernel="linear", decision_function_shape="ovo")
# svm_linear.fit(X_train, y_train)
# print("SVM训练模型评分：" + str(accuracy_score(y_train, svm_linear.predict(X_train))))
# print("SVM待测模型评分：" + str(accuracy_score(y_test, svm_linear.predict(X_test))))

model = RandomForestClassifier()  # 实例化模型RandomForestClassifier
model.fit(X_train, y_train)  # 在训练集上训练模型
print(model)  # 输出模型RandomForestClassifier

# 在测试集上测试模型
expected = y_test  # 测试样本的期望输出
predicted = model.predict(X_test)  # 测试样本预测

# 输出结果
print(metrics.classification_report(expected, predicted))  # 输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(expected, predicted))  # 混淆矩阵

auc = metrics.roc_auc_score(y_test, predicted)
accuracy = metrics.accuracy_score(y_test, predicted)  # 求精度
print("Accuracy: %.2f%%" % (accuracy * 100.0))
