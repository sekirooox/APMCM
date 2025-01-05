import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#　CLAHE改进的字节直方图算法
def clahe_rgb(image, clip_limit=2.0, tile_grid_size=(16, 16)):
    """
    对 RGB 图像应用 CLAHE 方法
    :param image: 输入 RGB 图像
    :param clip_limit: 对比度限制参数
    :param tile_grid_size: 区域划分大小
    :return: 增强后的 RGB 图像
    """
    # 转换为 LAB 颜色空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 分离 LAB 通道
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 对 L 通道应用 CLAHE
    l_channel_eq = clahe.apply(l_channel)

    # 合并增强后的 L 通道和原始 A、B 通道
    lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))

    # 转换回 RGB 颜色空间
    enhanced_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)
    return enhanced_image
# 灰度世界算法
def gray_world_algorithm(image):
    """
    改进的灰度世界算法
    """
    R_avg = np.mean(image[:, :, 0])
    G_avg = np.mean(image[:, :, 1])
    B_avg = np.mean(image[:, :, 2])
    Gray = (R_avg + G_avg + B_avg) / 3

    # 增加范围约束，避免过度偏移
    k_R = min(Gray / R_avg, 1.5)
    k_G = min(Gray / G_avg, 1.5)
    k_B = min(Gray / B_avg, 1.5)

    # 调整通道
    image[:, :, 0] = np.clip(image[:, :, 0] * k_R, 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * k_G, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * k_B, 0, 255)
    return image.astype(np.uint8)

# 暗通道先验法
def reverse_image(image):
    """
    图像反转
    :param image: 输入图像 (RGB格式)
    :return: 反转图像 R(x)
    """
    return 255 - image

def dark_channel(image, window_size=3):
    """
    计算暗通道
    :param image: 输入图像 (RGB格式)
    :param window_size: 局部窗口大小
    :return: 暗通道图像
    """
    min_channel = np.min(image, axis=2)  # 获取最小通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)  # 最小值滤波
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """
    估计大气光 A(x)
    :param image: 输入图像 (RGB格式)
    :param dark_channel: 暗通道图像
    :return: 大气光 A(x)
    """
    num_pixels = image.shape[0] * image.shape[1]
    top_pixels = 100  # 选择前100个亮点
    indices = np.unravel_index(np.argsort(-dark_channel.ravel()), dark_channel.shape)
    brightest = indices[0][:top_pixels], indices[1][:top_pixels]
    return np.mean(image[brightest], axis=0)

def estimate_transmission(reversed_image, atmospheric_light, omega=0.85, window_size=3):
    """
    改进透射率估计
    """
    atmospheric_light = np.clip(atmospheric_light, 1e-6, None)
    norm_image = reversed_image / atmospheric_light
    dark_channel_norm = dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel_norm
    return np.clip(transmission, 0.3, 0.95)  # 限制透射率范围

def recover_image(reversed_image, transmission, atmospheric_light, t_min=0.2):
    """
    改进的恢复图像
    """
    transmission = np.clip(transmission, t_min, 1)
    recovered = (reversed_image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return np.clip(recovered, 0, 255).astype(np.uint8)

# ---- 评价指标计算函数 ----
def calculate_psnr(original, enhanced):
    """
    计算 PSNR（峰值信噪比）
    :param original: 原始图像 (RGB格式)
    :param enhanced: 增强后的图像 (RGB格式)
    :return: PSNR 值
    """
    original = original.astype(np.float32)
    enhanced = enhanced.astype(np.float32)
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')  # 当两幅图像完全相同时，PSNR 无限大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_uciqe(image):
    """
    计算归一化后的 UCIQE（图像质量评价指标）
    :param image: 输入 RGB 图像，像素范围 [0, 255]
    :return: UCIQE 值
    """
    # 确保图像为 RGB 格式且归一化到 [0, 1]
    # image = image / 255.0

    # 转换到 LAB 色彩空间
    lab_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 1. 计算 L 通道的标准差
    l_std = np.std(l_channel)  # 归一化 L 通道计算标准差

    # 2. 计算色度 (chroma) 均值和标准差
    chroma = np.sqrt(a_channel.astype(np.float32)**2 + b_channel.astype(np.float32)**2) 
    chroma_mean = np.mean(chroma)
    chroma_std = np.std(chroma)

    # 3. 计算饱和度，防止除零
    saturation = chroma_std / (chroma_mean + 1e-6)

    # UCIQE 公式
    uciqe = 0.4680 * l_std + 0.2745 * chroma_mean + 0.2576 * saturation
    return uciqe


def calculate_uiqm(image):
    """
    计算归一化后的 UIQM（水下图像质量评价指标）
    :param image: 输入 RGB 图像，像素范围 [0, 255]
    :return: UIQM 值
    """
    # 确保图像为 RGB 格式且归一化到 [0, 1]
    # image = image / 255.0

    # 分离 RGB 通道
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # 1. 计算颜色对比 (UICM)
    uicm = np.abs(np.mean(r - g) + np.mean(r - b))

    # 2. 计算边缘强度 (UISM)
    uism = np.std(r) + np.std(g) + np.std(b)

    # 3. 计算颜色饱和度 (UIConM)
    uiconm = np.mean(np.sqrt((r - g)**2 + (r - b)**2))

    # UIQM 公式，权重为常量
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm


# ---- 图像增强算法（使用提供的函数） ----

def enhance_image(image, method):
    """根据方法名称调用增强函数"""
    if method == "blur":
        return clahe_rgb(image)
    elif method == "color cast":
        return gray_world_algorithm(image)
    elif method == "low light":
        # reversed_img = reverse_image(image)
        reversed_img=image
        dark = dark_channel(reversed_img)
        atmospheric_light = estimate_atmospheric_light(image, dark)
        transmission = estimate_transmission(reversed_img, atmospheric_light)
        return recover_image(reversed_img, transmission, atmospheric_light,t_min=0.3)
    else:# no degradation
        return image

# ---- 主处理逻辑 ----
def process_images(excel_file, image_folder, output_csv, output_folder):
    # 加载 Excel 文件
    classification_data = pd.read_excel(excel_file)

    # 创建保存增强图像的文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化存储结果的列表
    results = []

    for _, row in classification_data.iterrows():
        file_name = row['File Name']
        classifications = row['Classification'].split(", ")

        # 加载原始图像
        image_path = os.path.join(image_folder, file_name)
        original_image = cv2.imread(image_path)

        # 初始化最佳结果
        best_method = None
        best_psnr = -1
        best_image = None
        best_metrics = {}

        for method in classifications:
            # 判断是否处理模糊
            if method == "blur" and len(classifications) > 1:
                print(f"Skipping 'blur' for {file_name} as it has multiple classifications.")
                continue  # 跳过模糊的处理

            # 增强图像
            enhanced_image = enhance_image(original_image.copy(), method)

            # 保存增强后的图像
            # enhanced_file_name = f"{os.path.splitext(file_name)[0]}_{method}.png"  # 增加方法名称作为后缀
            # enhanced_image_path = os.path.join(output_folder, enhanced_file_name)
            # cv2.imwrite(enhanced_image_path, enhanced_image)

            # 打印增强方法和指标
            print(f"Processing File: {file_name}, Method: {method}")
            psnr = calculate_psnr(original_image, enhanced_image)
            uciqe = calculate_uciqe(enhanced_image)
            uiqm = calculate_uiqm(enhanced_image)
            print(f"PSNR: {psnr:.2f}, UCIQE: {uciqe:.2f}, UIQM: {uiqm:.2f}")

            # 更新最佳结果
            if psnr > best_psnr:
                best_psnr = psnr
                best_method = method
                best_image = enhanced_image
                best_metrics = {"PSNR": psnr, "UCIQE": uciqe, "UIQM": uiqm}

        # 保存结果
        if best_image is not None:
            result = {
                "File Name": file_name,
                "Best Method": best_method,
                "PSNR": best_metrics["PSNR"],
                "UCIQE": best_metrics["UCIQE"],
                "UIQM": best_metrics["UIQM"],
            }
            results.append(result)

    # 转换结果为 DataFrame 并保存为 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

# ---- 参数设置 ----

if __name__=='__main__':
    excel_file = 'classification_results2.xlsx'  # Excel 文件路径
    image_folder = 'Attachment/Attachment 2'  # 图像文件夹路径
    output_csv = 'metrics-result2.csv'
    output_folder='enhanced images'
    # 运行主函数
    process_images(excel_file, image_folder, output_csv,output_folder)
