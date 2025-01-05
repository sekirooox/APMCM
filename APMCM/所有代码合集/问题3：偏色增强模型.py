import cv2
import numpy as np
# 灰度世界算法
def gray_world_algorithm(image):
    """
    灰度世界算法实现
    :param image: 输入图像 (RGB格式)
    :return: 调整后的图像
    """
    # 计算 R、G、B 三个通道的均值
    R_avg = np.mean(image[:, :, 0])  # 红色通道均值
    G_avg = np.mean(image[:, :, 1])  # 绿色通道均值
    B_avg = np.mean(image[:, :, 2])  # 蓝色通道均值

    # 计算灰度值 Gray
    Gray = (R_avg + G_avg + B_avg) / 3

    # 计算增益系数
    k_R = Gray / R_avg
    k_G = Gray / G_avg
    k_B = Gray / B_avg

    # 对每个通道应用增益系数进行调整
    image[:, :, 0] = np.clip(image[:, :, 0] * k_R, 0, 255)  # 调整红色通道
    image[:, :, 1] = np.clip(image[:, :, 1] * k_G, 0, 255)  # 调整绿色通道
    image[:, :, 2] = np.clip(image[:, :, 2] * k_B, 0, 255)  # 调整蓝色通道

    return image.astype(np.uint8)

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
    img = cv2.imread("Attachment/Attachment 2/test_011.png") 
    cv2.imshow("Original Image", img)
    img_enhanced = gray_world_algorithm(img.copy())

    # cv2.imwrite("enhanced_image.jpg", img_enhanced)

    cv2.imshow("Adjusted Image", img_enhanced)
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
