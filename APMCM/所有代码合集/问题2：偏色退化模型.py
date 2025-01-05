import numpy as np
import cv2
def generate_color_cast_image(clear_image, depth_map, beta_values=(0.8, 0.1, 0.6), atmosphere_light=(0.4, 0.8, 0.3)):
    """
    Jaffe-McGlamery 模型生成退化的水下图像
    :param clear_image: 清晰图像 (H, W, C)，值范围 [0, 1]
    :param depth_map: 深度图 (H, W)，值范围 [0, 1]
    :param beta_values: 各通道的衰减系数 (R, G, B)
    :param atmosphere_light: 各通道的大气光 A_c (R, G, B)
    :return: 偏绿色的退化水下图像 (H, W, C)
    """
    degraded_image = np.zeros_like(clear_image)

    for c in range(3):  
        # 计算每个通道的透射率 t(x)
        t_channel = np.exp(-beta_values[c] * depth_map)

        degraded_image[..., c] = clear_image[..., c] * t_channel + atmosphere_light[c] * (1 - t_channel)
    
    return np.clip(degraded_image, 0, 1)

if __name__ == "__main__":

    clear_image = cv2.imread("Attachment/Attachment 1/image_015.png") / 255.0  # 读取清晰图像并归一化到 [0, 1]
    cv2.imshow('clear image',clear_image)

    height, width, _ = clear_image.shape

    # 生成深度图 (d(x))，例如使用线性梯度模拟
    # depth_map = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)  # 横向深度梯度
    depth_map = 1

    # 设置模型参数
    beta_values = (0.8, 0.2, 0.9)  # R, G, B 通道的衰减系数，绿色通道衰减较低
    atmosphere_light = (0.4, 0.8, 0.4)  # R, G, B 通道的大气光，绿色通道值较高

    degraded_image = generate_color_cast_image(clear_image, depth_map, beta_values, atmosphere_light)

    cv2.imshow('degraded_image',degraded_image)
    image_to_save = (degraded_image * 255).astype(np.uint8)  # 转换到 [0, 255] 范围
    cv2.imwrite('degraded_image.png', image_to_save)
    cv2.waitKey(0)