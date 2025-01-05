import cv2
import numpy as np

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
    recovered_img = recover_image(reversed_img, transmission, atmospheric_light)

    # 显示结果
    cv2.imshow("Original Image", img)
    # cv2.imshow("Reversed Image", reversed_img)
    # cv2.imshow("Dark Channel", dark)
    cv2.imshow("Recovered Image", recovered_img)
    cv2.imwrite("recovered_image.png", recovered_img)  # 保存恢复图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
