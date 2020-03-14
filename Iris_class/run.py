import numpy as np  # 計算用
import matplotlib.pyplot as plt  # 描画用
import pandas as pd  # 行列計算、csv読み込み
# import mglearn

x = np.linspace(-10, 10, 100)

y = np.sin(x)

plt.plot(x, y, marker="x")

plt.show()
