import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.measure import shannon_entropy
import csv
import os

from sklearn.mixture import GaussianMixture

def load_image(image_path,require_gray_image=False,require_show=True):
    """
    加载图片
    :param image_path: 图片路径
    :return: 图片的 numpy 数组
    """
    image = cv2.imread(image_path)  
    if require_gray_image:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if require_show:
        cv2.imshow(image_path,image)

    return image
"""
偏色:RGB直方图,全局饱和度分析的直方图，局部饱和度分析的热力图
"""
"""以下是偏色部分的图像特征提取和分析"""
def calculate_rgb_histogram(image_path):
    """
    计算 RGB 三个通道的直方图
    :param image: 输入图片
    :return: 三个通道的直方图
    """
    image=load_image(image_path,require_gray_image=False,require_show=False)
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])  # 红色通道
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # 绿色通道
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])  # 蓝色通道
    return hist_r, hist_g, hist_b
# 绘制rgb直方图
def plot_rgb_histogram(hist_r, hist_g, hist_b):

    """
    绘制 RGB 三通道的直方图
    :param hist_r: 红色通道直方图
    :param hist_g: 绿色通道直方图
    :param hist_b: 蓝色通道直方图
    """
    plt.figure(figsize=(8, 6))
    plt.plot(hist_r, color='red', label='Red Channel',linestyle='--')  # 红色通道
    plt.plot(hist_g, color='green', label='Green Channel',linestyle='--')  # 绿色通道
    plt.plot(hist_b, color='blue', label='Blue Channel',linestyle='--')  # 蓝色通道
    plt.title('RGB Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')  # 图例
    plt.grid(alpha=0.3)
    plt.show()

# 计算全局饱和度直方图
def calculate_saturation_histogram(image):
    """
    计算全局饱和度直方图
    return:ndarray
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 转换为 HSV

    saturation = image_hsv[:, :, 1]  

    return saturation.flatten()
# 绘制全局饱和度的直方图
def plot_saturation_histogram(saturation_array):
    """
    全局饱和度分析：计算饱和度直方图并可视化
    :param image_path: 图像路径
    """

    # 绘制饱和度直方图
    plt.figure(figsize=(8, 6))
    plt.hist(saturation_array, bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title("Global Saturation Histogram")
    plt.xlabel("Saturation Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()

    # 统计分析
    mean_saturation = np.mean(saturation_array)
    std_saturation = np.std(saturation_array)
    print(f"Mean Saturation: {mean_saturation:.2f}")
    print(f"Standard Deviation of Saturation: {std_saturation:.2f}")
# 绘制局部饱和度的热力图
def plot_local_saturation_heatmap(image_path, grid_size=8):
    """
    局部饱和度分析：将图像分割为网格，计算每个网格的平均饱和度，并绘制热力图。
    :param image_path: 图像路径
    :param grid_size: 网格的大小（例如，4表示将图像分割为4x4的网格）
    """
    image = cv2.imread(image_path)   
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  

    saturation = image_hsv[:, :, 1]  

    h, w = saturation.shape

    grid_h, grid_w = h // grid_size, w // grid_size
    local_means = []

    for i in range(grid_size):
        for j in range(grid_size):
            grid = saturation[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            local_means.append(np.mean(grid))

    # 将局部饱和度均值转化为矩阵形式，便于绘制热力图
    local_means = np.array(local_means).reshape((grid_size, grid_size))

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(local_means, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label="Mean Saturation")
    plt.title("Local Saturation Analysis")
    plt.show()
# 计算高频分量占比，低于阈值则可能存在模糊现象
def calculate_high_frequency_energy(image_path, radius=50):
    """
    计算高频分量能量占比
    :param image_path: 图片路径
    :param radius: 低频掩模的半径
    :return: 高频能量占比
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    total_energy = np.sum(np.abs(fft_shift))

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)  # 低频区域设为 0

    fft_high_freq = fft_shift * mask
    high_freq_energy = np.sum(np.abs(fft_high_freq))

    # 计算高频能量占比
    high_freq_ratio = high_freq_energy / total_energy

    return high_freq_ratio

# 绘制高频分量并可视化
def plot_high_frequency_component(image_path):
    """
    :param image_path: 图片路径
    :return image: 高频分量图像
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 进行傅里叶变换 (FFT)
    fft = np.fft.fft2(image)  
    fft_shift = np.fft.fftshift(fft)  

    # 计算频谱图
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    # 提取高频分量
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  
    mask = np.ones((rows, cols), np.uint8)
    r = 50  
    cv2.circle(mask, (ccol, crow), r, 0, -1) 
    fft_shift_high_freq = fft_shift * mask

    # 逆傅里叶变换恢复高频分量图像
    fft_high_freq = np.fft.ifftshift(fft_shift_high_freq)
    high_freq_image = np.abs(np.fft.ifft2(fft_high_freq))
    # print(magnitude_spectrum.shape)
    # print(high_freq_image.shape)
    # 可视化
    plt.figure(1,figsize=(12, 8))
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.figure(2,figsize=(12, 8))
    plt.title("Frequency Spectrum")
    plt.imshow(magnitude_spectrum, cmap='gray')

    plt.figure(3,figsize=(12,8))
    plt.title("High Frequency Image")
    plt.imshow(high_freq_image, cmap='gray')

    plt.show()

# 绘制边缘检测，如果边缘像素的占比比较低，则说明图像模糊
def calculate_edge_detection(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, 30, 180)  # 阈值范围可以调整

    # 统计边缘像素数量
    edge_pixel_count = np.sum(edges > 0)
    total_pixel_count = gray_image.size
    edge_ratio = edge_pixel_count / total_pixel_count * 100  # 边缘像素占比（百分比）

    return edge_ratio

def plot_edge_detection(image_path, threshold=5):
    """
    基于边缘检测分析图像模糊程度
    :param image_path: 图像路径
    :param threshold: 判断边缘像素数量的阈值
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, 40, 180)  # 阈值范围可以调整

    # 统计边缘像素数量
    edge_pixel_count = np.sum(edges > 0)
    total_pixel_count = gray_image.size
    edge_ratio = edge_pixel_count / total_pixel_count * 100  # 边缘像素占比（百分比）

    edge_strength_mean = np.mean(edges[edges > 0]) 
    edge_strength_std = np.std(edges[edges > 0])  

    plt.figure(1,figsize=(12, 6))
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.figure(2,figsize=(12, 6))
    plt.title("Edge Detection (Canny)")
    plt.imshow(edges, cmap='gray')
    plt.show()




def calculate_gray_histogram(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  
    hist = hist.flatten()  
    return hist
# 绘制灰度直方图，如果低灰度值占比大于某一个阈值，则为弱光
def plot_gray_histogram(hist):

    """
    绘制灰度直方图，用于分析水下图像的亮度分布
    """

    plt.figure(figsize=(8, 6))
    plt.title("Gray Level Histogram")
    plt.xlabel("Pixel Intensity (Gray Level)")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black', label='Gray Level Distribution')
    plt.fill_between(range(256), hist, color='gray', alpha=0.6)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # mean_gray = np.mean(gray_image)
    # std_gray = np.std(gray_image)
    # low_light_ratio = np.sum(gray_image < 50) / gray_image.size * 100  # 阈值低于50的像素比例

    # print(f"Mean Gray Level: {mean_gray:.2f}")
    # print(f"Standard Deviation of Gray Level: {std_gray:.2f}")
    # print(f"Low Light Pixel Ratio : {low_light_ratio:.2f}%")
# 计算灰度图的熵，熵越低，说明图像越暗(这个方法不是很靠谱)
def calculate_image_entropy(image_path):
    """
    计算图像的熵并分析是否为弱光图像
    :param image_path: 图像路径
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    skimage_entropy = shannon_entropy(gray_image)
    # print(f"Computed Entropy: {skimage_entropy:.2f}")
    return skimage_entropy
# 灰度热力图
def plot_gray_heatmap(image_path):
    """
    绘制灰度热力图
    :param image_path: 图片路径
    """
    # 加载灰度图像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(gray_image, cmap="hot", interpolation="nearest")
    plt.title("Gray Level Heatmap")
    plt.colorbar(label="Gray Value")
    plt.show()


def process_image_to_feature(image_path):
    """
    从图片中提取特征向量，并对特定特征类型进行独立处理
    :param image_path: 图片路径
    :return: 特征向量 (1D numpy array)
    """
    hist_r, hist_g, hist_b = calculate_rgb_histogram(image_path)

    edge_ratio = calculate_edge_detection(image_path=image_path)


    gray_histogram = calculate_gray_histogram(image_path=image_path)

    image_entropy = calculate_image_entropy(image_path=image_path)

    # 对 RGB 直方图和灰度直方图进行标准化
    hist_r = hist_r / np.sum(hist_r) if np.sum(hist_r) != 0 else hist_r
    hist_g = hist_g / np.sum(hist_g) if np.sum(hist_g) != 0 else hist_g
    hist_b = hist_b / np.sum(hist_b) if np.sum(hist_b) != 0 else hist_b
    gray_histogram = gray_histogram / np.sum(gray_histogram) if np.sum(gray_histogram) != 0 else gray_histogram

    # 对边缘比例进行范围缩放（0~1）
    edge_ratio = np.clip(edge_ratio, 0, 1)  # 确保比例在合理范围内


    # 拼接最终特征向量
    feature_vector = np.concatenate([
        hist_r.reshape(-1), 
        hist_g.reshape(-1), 
        hist_b.reshape(-1), 
        np.array([edge_ratio]), 
        gray_histogram, 
        np.array([image_entropy])
    ])

    return feature_vector
def extract_features_from_folder(folder_path):
    """
    从文件夹中的所有图片提取特征
    :param folder_path: 文件夹路径
    :return: 特征矩阵 (n_samples, n_features)，图片路径列表
    """
    features = []
    image_paths = []

    # 遍历文件夹中的图片
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            feature_vector = process_image_to_feature(image_path)
            features.append(feature_vector)
            image_paths.append(image_path)
    np.save('features.npy',np.array(features))
    return np.array(features), image_paths
def train_gmm_model(features, n_components=3):
    """
    使用 GMM 模型对特征矩阵进行聚类
    :param features: 特征矩阵 (n_samples, n_features)
    :param n_components: GMM 模型中的类别数
    :return: 训练好的 GMM 模型
    """

    # 训练 GMM 模型
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)

    return gmm
def classify_images_with_gmm(gmm,features, image_paths, threshold=0.6, output_csv="classification_results.csv"):
    """
    使用 GMM 模型对图片进行分类，设定概率阈值，支持多分类，并将结果保存为 CSV 文件
    :param gmm: 训练好的 GMM 模型
    :param scaler: 特征标准化器
    :param features: 特征矩阵 (n_samples, n_features)
    :param image_paths: 图片路径列表
    :param threshold: 分类概率阈值，高于该值则认为符合分类
    :param output_csv: 输出的 CSV 文件路径
    """
    predictions = gmm.predict(features)
    probabilities = gmm.predict_proba(features)

    classification_results = []

    for i, path in enumerate(image_paths):

        image_name = path.split("/")[-1] if "/" in path else path.split("\\")[-1]
        
        classified_labels = []
        for class_idx, prob in enumerate(probabilities[i]):
            if prob >= threshold:
                classified_labels.append(f"Class {class_idx} (Prob: {prob:.2f})")

        if not classified_labels:
            classified_labels.append("Unclassified")

        classification_results.append([image_name, "; ".join(classified_labels)])
        
        print(f"Image: {image_name}")
        print(f"  Predicted Class: {predictions[i]}")
        print(f"  Probabilities: {probabilities[i]}")
        print(f"  Classification: {classified_labels}\n")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Classification Result"])
        writer.writerows(classification_results)

    print(f"Classification results saved to {output_csv}")

def kl_divergence(p, q):
    """
    计算 KL 散度 D_KL(P || Q)
    :param p: 概率分布 P
    :param q: 概率分布 Q
    :return: KL 散度
    """
    epsilon = 1e-10  # 防止分母为零
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))
def calculate_kl_for_channels(hist_r, hist_g, hist_b):
    """
    计算 RGB 通道之间的 KL 散度
    :param hist_r: 红色通道直方图
    :param hist_g: 绿色通道直方图
    :param hist_b: 蓝色通道直方图
    :return: KL 散度值 (R||G, R||B, G||B)
    """
    # 将直方图归一化为概率分布
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)

    # 计算 KL 散度
    kl_rg = kl_divergence(hist_r, hist_g)
    kl_rb = kl_divergence(hist_r, hist_b)
    kl_gb = kl_divergence(hist_g, hist_b)

    return kl_rg, kl_rb, kl_gb

# 使用KL散度定量判定是否偏色
def analyze_image_color_distribution(image_path, threshold=1):
    """
    根据 KL 散度分析图像颜色分布是否存在失色问题
    :param image_path: 图片路径
    :param threshold: 判断失色的 KL 散度阈值
    :return: 分析结果
    """
    hist_r, hist_g, hist_b = calculate_rgb_histogram(image_path)

    kl_rg, kl_rb, kl_gb = calculate_kl_for_channels(hist_r, hist_g, hist_b)

    print(f"KL Divergence (R||G): {kl_rg:.4f}")
    print(f"KL Divergence (R||B): {kl_rb:.4f}")
    print(f"KL Divergence (G||B): {kl_gb:.4f}")

    if kl_rg > threshold or kl_rb > threshold or kl_gb > threshold:
        print("Potential color imbalance or loss detected.")
    else:
        print("Color distribution appears balanced.")

# 饱和度判定
def analyze_saturation_distribution(saturation, low_threshold=50, high_threshold=200):
    """
    分析饱和度分布，计算统计特性
    :param saturation: 饱和度值数组
    :param low_threshold: 判断低饱和度的阈值
    :param high_threshold: 判断高饱和度的阈值
    :return: 饱和度统计信息
    """
    # 计算均值和标准差
    mean_saturation = np.mean(saturation)
    std_saturation = np.std(saturation)

    # 计算低饱和度和高饱和度比例
    low_saturation_ratio = np.sum(saturation < low_threshold) / len(saturation) * 100
    high_saturation_ratio = np.sum(saturation > high_threshold) / len(saturation) * 100

    return mean_saturation, std_saturation, low_saturation_ratio, high_saturation_ratio
def analyze_image_saturation(image_path, low_threshold=50, high_threshold=200):
    """
    通过饱和度分析是否偏色
    :param image_path: 图片路径
    :param low_threshold: 判断低饱和度的阈值
    :param high_threshold: 判断高饱和度的阈值
    """
    # 加载图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算饱和度直方图
    saturation = calculate_saturation_histogram(image)

    # 分析饱和度统计特性
    mean_saturation, std_saturation, low_saturation_ratio, high_saturation_ratio = analyze_saturation_distribution(
        saturation, low_threshold, high_threshold
    )

    print(f"Mean Saturation: {mean_saturation:.2f}")
    print(f"Standard Deviation of Saturation: {std_saturation:.2f}")
    print(f"Low Saturation Ratio (<{low_threshold}): {low_saturation_ratio:.2f}%")
    print(f"High Saturation Ratio (>{high_threshold}): {high_saturation_ratio:.2f}%")

    # 判断偏色问题
    if low_saturation_ratio > 50:
        print("The image may be desaturated (low color purity).")
    elif high_saturation_ratio > 50:
        print("The image may have oversaturated regions (high color intensity).")
    else:
        print("The image has balanced color saturation.")

# 高频图像判定
def analyze_image_blur(image_path, threshold=0.1, radius=50):
    """
    根据高频能量占比分析图像是否模糊
    :param image_path: 图片路径
    :param threshold: 判断模糊的高频能量占比阈值
    :param radius: 低频掩模的半径
    """
    # 计算高频能量占比
    high_freq_ratio = calculate_high_frequency_energy(image_path, radius)

    # 打印分析结果
    print(f"High Frequency Energy Ratio: {high_freq_ratio:.4f}")

    if high_freq_ratio < threshold:
        print("The image is likely blurred.")
    else:
        print("The image appears to be sharp.")

# 边缘检测判定
def analyze_edge_detection(image_path,threshold=5):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, 40, 180)  # 阈值范围可以调整
    
    # 统计边缘像素数量
    edge_pixel_count = np.sum(edges > 0)
    total_pixel_count = gray_image.size
    edge_ratio = edge_pixel_count / total_pixel_count * 100  # 边缘像素占比（百分比）

    edge_strength_mean = np.mean(edges[edges > 0]) 
    edge_strength_std = np.std(edges[edges > 0])  

    print(f"Edge Pixel Count: {edge_pixel_count}")
    print(f"Edge Pixel Ratio: {edge_ratio:.2f}%")
    print(f"Edge Strength Mean: {edge_strength_mean:.2f}")
    print(f"Edge Strength Std: {edge_strength_std:.2f}")

    if edge_ratio < threshold:
        print("The image is likely blurred.")
    else:
        print("The image is likely sharp.")

# 灰度图判定
def analyze_gray_image(image_path, low_threshold=50):
    """
    分析灰度图像的弱光特性
    :param image_path: 图片路径
    :param low_threshold: 判断低灰度的阈值
    :return: 灰度均值, 标准差, 低灰度像素占比
    """
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    gray_mean = np.mean(gray_image)
    gray_std = np.std(gray_image)

    low_gray_ratio = np.sum(gray_image < low_threshold) / gray_image.size * 100

    print(f"Gray Mean: {gray_mean:.2f}")
    print(f"Gray Std: {gray_std:.2f}")
    print(f"Low Gray Ratio (<{low_threshold}): {low_gray_ratio:.2f}%")

    # 判断弱光
    if gray_mean < 80 and low_gray_ratio > 60:
        print("The image is likely underexposed (low light).")
    else:
        print("The image appears to have balanced brightness.")

# 熵值判定
def analyze_entropy(image_path,threshold=400):
    image_entropy=calculate_image_entropy(image_path=image_path)# 1
    image_entropy=np.exp(image_entropy)# 放大
    print(f'The image entropy of exponent is {image_entropy}')
    if image_entropy<threshold:
        print("The image is likely underexposed (low light).")
    else:
        print("The image appears to have balanced brightness.")

def classify_image(image_path):
    """
    对单张图片进行综合判定，返回分类结果
    :param image_path: 图片路径
    :return: 文件名和分类结果（字符串，用逗号隔开）
    """
    categories = []  # 存储分类结果
    file_name = os.path.basename(image_path)  # 获取文件名

    # KL 散度偏色判定
    try:
        hist_r, hist_g, hist_b = calculate_rgb_histogram(image_path)
        kl_rg, kl_rb, kl_gb = calculate_kl_for_channels(hist_r, hist_g, hist_b)
        if kl_rg > 1 or kl_rb > 1 or kl_gb > 1:
            categories.append("color cast")
    except Exception as e:
        print(f"Error in KL divergence analysis for {file_name}: {e}")

    # 饱和度偏色判定
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        saturation = calculate_saturation_histogram(image)
        mean_saturation, std_saturation, low_saturation_ratio, high_saturation_ratio = analyze_saturation_distribution(
            saturation
        )
        if low_saturation_ratio > 50:
            categories.append("color cast")
        elif high_saturation_ratio > 50:
            categories.append("color cast")
    except Exception as e:
        print(f"Error in saturation analysis for {file_name}: {e}")

    # 高频分量模糊判定
    try:
        high_freq_ratio = calculate_high_frequency_energy(image_path, radius=50)
        if high_freq_ratio < 0.1:
            categories.append("blur")
    except Exception as e:
        print(f"Error in high frequency analysis for {file_name}: {e}")

    # 边缘检测模糊判定
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 40, 180)
        edge_pixel_count = np.sum(edges > 0)
        total_pixel_count = gray_image.size
        edge_ratio = edge_pixel_count / total_pixel_count * 100
        if edge_ratio < 5:
            categories.append("blur")
    except Exception as e:
        print(f"Error in edge detection analysis for {file_name}: {e}")

    # 灰度图弱光判定
    try:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gray_mean = np.mean(gray_image)
        low_gray_ratio = np.sum(gray_image < 50) / gray_image.size * 100
        if gray_mean < 80 and low_gray_ratio > 60:
            categories.append("low light")
    except Exception as e:
        print(f"Error in gray analysis for {file_name}: {e}")
    # 熵值判定
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_entropy=calculate_image_entropy(image_path=image_path)# 1
        image_entropy=np.exp(image_entropy)# 放大
        if image_entropy<400:
            categories.append('low light')
    except Exception as e:
        print(f"Error in gray analysis for {file_name}: {e}")

    # 未检测到退化现象
    if not categories:
        categories.append("no degradation")

    return file_name, ", ".join(sorted(set(categories)))

def classify_images_in_folder(folder_path, output_excel_path):
    """
    对文件夹中的所有图片进行分类并保存结果到 Excel
    :param folder_path: 图片文件夹路径
    :param output_excel_path: 输出 Excel 文件路径
    """
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            file_name, categories = classify_image(image_path)
            results.append({"File Name": file_name, "Classification": categories})

    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"Classification results saved to {output_excel_path}")





if __name__ == "__main__":

    image_path = "Attachment/Attachment 1/image_020.png"
    require_gray_image=False
    require_show=True
    image = load_image(image_path,require_gray_image=require_gray_image,require_show=require_show)
    # RGB分布检测偏色
    # analyze_image_color_distribution(image_path=image_path)
    # hist_r, hist_g, hist_b = calculate_rgb_histogram(image_path)
    # plot_rgb_histogram(hist_r, hist_g, hist_b)# 256,)
    # kl_rg, kl_rb, kl_gb=calculate_kl_for_channels(hist_r,hist_g,hist_b)


    # 饱和度检测偏色
    saturation_histogram = calculate_saturation_histogram(image)# image.shape*3(RGB)
    plot_saturation_histogram(saturation_histogram)
    # analyze_image_saturation(image_path=image_path)
    # plot_local_saturation_heatmap(image_path, grid_size=8)

    # 高频检测模糊
    # analyze_image_blur(image_path=image_path,threshold=0.1,radius=150)# 检测明显的模糊
    # plot_high_frequency_component(image_path=image_path)# image282,jpg

    # 边缘检测模糊
    # edge_ratio=calculate_edge_detection(image_path=image_path)
    # analyze_edge_detection(image_path=image_path,threshold=5)
    # plot_edge_detection(image_path=image_path)

    # 灰度图弱光检测
    # gray_histogram=calculate_gray_histogram(image_path=image_path)# 256,)
    # plot_gray_histogram(gray_histogram)
    # plot_gray_heatmap(image_path=image_path)
    # analyze_gray_image(image_path=image_path,low_threshold=50)


    # 熵值检验
    # image_entropy=calculate_image_entropy(image_path=image_path)# 1
    # analyze_entropy(image_path=image_path,threshold=400)

    
    # 综合检验
    folder_path = "Attachment/Attachment 2"  
    output_excel_path = "classification_results2.xlsx"  
    classify_images_in_folder(folder_path, output_excel_path)

    # cv2.waitKey(0)