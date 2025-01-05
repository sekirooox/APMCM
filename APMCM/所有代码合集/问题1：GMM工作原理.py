import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义不同的均值、标准差和高度缩放因子
params = [
    {"mean": 0, "std_dev": 1, "scale_factor": 1.5},  # 主分布参数
    {"mean": 2, "std_dev": 1.2, "scale_factor": 1.2},  # 子分布参数
    {"mean": -1, "std_dev": 1.5, "scale_factor": 1.0},  # 子分布参数
    {"mean": 1, "std_dev": 0.8, "scale_factor": 0.8},  # 子分布参数
]

# 定义 x 轴范围
x = np.linspace(-5, 5, 1000)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制每条高斯曲线
for i, param in enumerate(params):
    mean = param["mean"]  # 获取均值
    std_dev = param["std_dev"]  # 获取标准差
    scale_factor = param["scale_factor"]  # 获取缩放因子

    # 计算高斯分布的概率密度函数
    y = scale_factor * norm.pdf(x, loc=mean, scale=std_dev)
    
    # 绘制主分布（红色，加粗线条，带填充）
    if i == 0:  # 主分布
        plt.plot(x, y, label=f"Main Distribution", color="red", linewidth=3)
        plt.fill_between(x, y, color="red", alpha=0.3)  # 填充主分布下方区域
    else:  # 子分布
        plt.plot(x, y, label=f"Sub Distribution {i}", linewidth=1.5)


# 添加标题和标签
plt.title("Gaussian Distributions with Different Heights", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Scaled Probability Density", fontsize=12)

# 显示图例
plt.legend()

# 显示图形
plt.show()
