import cv2
import numpy as np
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

def estimate_transmission(reversed_image, atmospheric_light, omega=0.8, window_size=3):
    """
    估计透射率 t(x)
    :param reversed_image: 反转图像
    :param atmospheric_light: 大气光 A(x)
    :param omega: 衰减系数
    :param window_size: 局部窗口大小
    :return: 透射率 t(x)
    """
    norm_image = reversed_image / atmospheric_light  # 按通道归一化
    dark_channel_norm = dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel_norm
    return transmission

def recover_image(reversed_image, transmission, atmospheric_light, t_min=0.4):
    """
    恢复图像
    :param reversed_image: 反转图像
    :param transmission: 透射率 t(x)
    :param atmospheric_light: 大气光 A(x)
    :param t_min: 最小透射率
    :return: 恢复的清晰图像 J(x)
    """
    transmission = np.clip(transmission, t_min, 1)  # 限制透射率范围
    recovered = (reversed_image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return np.clip(recovered, 0, 255).astype(np.uint8)
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
    # 加载输入图像
    img = cv2.imread("Attachment/Attachment 1/image_020.png")  # 替换为你的图像路径

    # 1. 图像反转
    # reversed_img = reverse_image(img)
    reversed_img=img

    # 2. 暗通道计算
    dark = dark_channel(reversed_img)

    # 3. 大气光估计
    atmospheric_light = estimate_atmospheric_light(reversed_img, dark)

    # 4. 透射率估计
    transmission = estimate_transmission(reversed_img, atmospheric_light, omega=0.8)

    # 5. 图像恢复
    img_enhanced = recover_image(reversed_img, transmission, atmospheric_light)

    # 显示结果
    cv2.imshow("Original Image", img)
    # cv2.imshow("Reversed Image", reversed_img)
    # cv2.imshow("Dark Channel", dark)
    cv2.imshow("Recovered Image", img_enhanced)

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