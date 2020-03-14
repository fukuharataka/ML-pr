import numpy as np  # 計算用
import matplotlib.pyplot as plt  # 描画用
import pandas as pd  # 行列計算、csv読み込み
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("y_train_shape: {}".format(y_train.shape))

print("X_test_shape: {}".format(X_test.shape))
print("y_test_shape: {}".format(y_test.shape))
