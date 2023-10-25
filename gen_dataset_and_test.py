import pandas as pd
from sklearn.model_selection import train_test_split  # 划分数据集与测试集
from sklearn import metrics  # 分类结果评价函数
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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

# add labels
tdata["is_https"] = 1
fdata["is_https"] = 0

# gen dataset
dataset = pd.concat([tdata, fdata])

# --------------------------------preprocessing

# define features of number type
number_features_name = [
    "port",
    "cipher",
    "server_hello_length",
    "server_hello_extensions_length",
]

# port as features?
if not args.withPort:
    dataset.drop(columns="port", inplace=True)
    number_features_name.remove("port")

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
    dataset = dataset[dataset["protocol"] == protocol]
    dataset.drop(columns="protocol", inplace=True)
else:
    # onehot encode
    dataset["is_tls1.0"] = dataset["protocol"].apply(lambda x: 1 if x == 769 else 0)
    dataset["is_tls1.1"] = dataset["protocol"].apply(lambda x: 1 if x == 770 else 0)
    dataset["is_tls1.2"] = dataset["protocol"].apply(lambda x: 1 if x == 771 else 0)
    dataset["is_tls1.3"] = dataset["protocol"].apply(lambda x: 1 if x == 772 else 0)
    dataset.drop(columns="protocol", inplace=True)


# standard number type features and return to dataframe
number_features = dataset[number_features_name]
std_transfer = StandardScaler()
number_features = pd.DataFrame(
    std_transfer.fit_transform(number_features), columns=number_features_name
)

dataset.drop(labels=number_features_name, axis=1, inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset = pd.concat([number_features, dataset], axis=1)

# ---------------------training
train = dataset.drop(["is_https"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    train,
    dataset["is_https"],
    random_state=1,
    train_size=0.8,
    stratify=dataset["is_https"],
)

model = RandomForestClassifier()  # 实例化模型RandomForestClassifier
model.fit(X_train, y_train)  # 在训练集上训练模型
print(model)  # 输出模型RandomForestClassifier

expected = y_test
predicted = model.predict(X_test)

# 输出结果
print(metrics.classification_report(expected, predicted))  # 输出结果，精确度、召回率、f-1分数
print(metrics.confusion_matrix(expected, predicted))  # 混淆矩阵

auc = metrics.roc_auc_score(y_test, predicted)
accuracy = metrics.accuracy_score(y_test, predicted)  # 求精度
print("Accuracy: %.2f%%" % (accuracy * 100.0))
