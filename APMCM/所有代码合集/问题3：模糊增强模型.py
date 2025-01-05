import cv2
import numpy as np
import matplotlib.pyplot as plt

def clahe_rgb(image, clip_limit=2.0, tile_grid_size=(8, 8)):
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


if __name__ == "__main__":
    # 加载 RGB 图像
    img = cv2.imread("Attachment/Attachment 1/image_282.jpg")  # 替换为你的图像路径
    
    # 应用 CLAHE 方法
    clip_limit=1.0 # 与对比度正相关
    tile_grid_size=(32,32)# 越大计算资源越耗费，但是细节更加明显
    img_enhanced = clahe_rgb(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # 显示原始图像和增强图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Enhanced Image", img_enhanced)

    # 可选保存增强后的图像
    cv2.imwrite("enhanced_image.jpg", img_enhanced)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 计算图像增强前后的指标
    psnr_value = calculate_psnr(img, img_enhanced)
    uciqe_original = calculate_uciqe(img)
    uciqe_enhanced = calculate_uciqe(img_enhanced)
    uiqm_original = calculate_uiqm(img)
    uiqm_enhanced = calculate_uiqm(img_enhanced)

    print(f"PSNR between original and enhanced image: {psnr_value:.2f}")
    print(f"UCIQE for original image: {uciqe_original:.4f}")
    print(f"UCIQE for enhanced image: {uciqe_enhanced:.4f}")
    print(f"UIQM for original image: {uiqm_original:.4f}")
    print(f"UIQM for enhanced image: {uiqm_enhanced:.4f}")
