import cv2
import numpy as np
import os

def generate_blur_kernel(kernel_size=15, sigma=3):
    """
    生成高斯模糊核，用于模拟散射效应
    :param kernel_size: 模糊核大小（必须为奇数）
    :param sigma: 高斯核标准差
    :return: 高斯模糊核
    """
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def apply_blur(image, kernel):
    """
    对图像应用模糊（卷积）
    :param image: 输入图像 (H, W, C)
    :param kernel: 模糊核 (k, k)
    :return: 模糊图像
    """
    blurred = np.zeros_like(image)
    for c in range(3):  # 对每个通道分别应用卷积
        blurred[..., c] = cv2.filter2D(image[..., c], -1, kernel)
    return blurred

def generate_underwater_blur_image(clear_image, depth_map, beta=0.5, background_light=0.2, kernel_size=15, sigma=3):
    """
    使用模糊退化模型生成水下模糊图像
    :param clear_image: 清晰图像 J(x) (H, W, C)，范围 [0, 1]
    :param depth_map: 深度图 d(x) (H, W)，范围 [0, 1]
    :param beta: 光衰减系数
    :param background_light: 背景光强度 B
    :param kernel_size: 高斯模糊核大小
    :param sigma: 高斯核标准差
    :return: 模糊退化图像 I(x) (H, W, C)
    """
    # 计算透射率 t(x)
    transmission = np.exp(-beta * depth_map)

    # 生成高斯模糊核并对清晰图像进行模糊处理
    blur_kernel = generate_blur_kernel(kernel_size, sigma)
    blurred_image = apply_blur(clear_image, blur_kernel)

    # 生成退化图像
    degraded_image = blurred_image * transmission[..., None] + background_light * (1 - transmission[..., None])
    return np.clip(degraded_image, 0, 1)

if __name__ == "__main__":
    # 加载清晰图像
    clear_image_path = "Attachment/Attachment 1/image_015.png"
    clear_image = cv2.imread(clear_image_path)

    if clear_image is None:
        print(f"Error: Unable to load image from {clear_image_path}")
        exit()

    clear_image = clear_image / 255.0  # 归一化到 [0, 1]

    # 模拟深度图
    height, width, _ = clear_image.shape
    depth_map = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)

    # 模糊退化模型参数
    beta = 0.7  # 光衰减系数
    background_light = 0.1  # 环境光强度
    kernel_size = 21  # 模糊核大小
    sigma = 5  # 模糊强度

    # 生成模糊退化图像
    underwater_blur_image = generate_underwater_blur_image(clear_image, depth_map, beta, background_light, kernel_size, sigma)

    # 保存结果
    output_folder = "Underwater_Blur_Images"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "image_015_underwater_blur.jpg")
    cv2.imwrite(output_path, (underwater_blur_image * 255).astype(np.uint8))

    print(f"Underwater blur image saved to {output_path}")

    # 显示结果
    cv2.imshow("Original Image", clear_image)
    cv2.imshow("Underwater Blur Image", underwater_blur_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
