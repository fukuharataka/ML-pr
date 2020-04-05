import numpy as np  # 計算用
import matplotlib.pyplot as plt  # 描画用
import pandas as pd  # 行列計算、csv読み込み
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)

print("^^^^^^^^^^^^^^^^")
print("Prediction: {}".format(prediction))
print("Predicted target name : {}".format(
    iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions: \n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

# pd.plotting.scatter_matrix(
#     iris_dataframe,
#     c=y_train,
#     figsize=(
#         15,
#         15),
#     marker='o',
#     hist_kwds={
#         'bins': 20},
#     s=60,
#     alpha=.8,
#     cmap=mglearn.cm3)

# plt.show()
