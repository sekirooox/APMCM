import numpy as np
import cv2

def generate_low_light_image(clear_image, depth_map, beta=0.3, atmosphere_light=(0.1, 0.1, 0.1)):
    """
    基于弱光的 Jaffe-McGlamery 模型生成退化的水下图像
    :param clear_image: 清晰图像 (H, W, C)，值范围 [0, 1]
    :param depth_map: 深度图 (H, W)，值范围 [0, 1]
    :param beta: 衰减系数，控制光强的快速衰减
    :param atmosphere_light: 大气光 A_c (R, G, B)，用于模拟弱光场景
    :return: 弱光退化的水下图像 (H, W, C)
    """
    transmission = np.exp(-beta * depth_map)

    atmosphere_light = np.array(atmosphere_light).reshape(1, 1, 3)

    degraded_image = clear_image * transmission[..., None] + atmosphere_light * (1 - transmission[..., None])
    
    # 确保结果在 [0, 1] 范围内
    return np.clip(degraded_image, 0, 1)


if __name__=='__main__':
    clear_image = cv2.imread("Attachment/Attachment 1/image_015.png") / 255.0  # 读取清晰图像并归一化到 [0, 1]
    height, width, _ = clear_image.shape
    cv2.imshow('clear image',clear_image)

    # 生成深度图 (d(x))，例如使用线性梯度模拟
    # depth_map = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)  # 横向深度梯度
    depth_map = 1

    beta = 0.2  # 较高的衰减系数，模拟快速光衰减
    atmosphere_light = (0.01, 0.01, 0.01)  # 低值大气光，模拟弱光场景


    degraded_image = generate_low_light_image(clear_image, depth_map, beta, atmosphere_light)

    # 保存结果
    cv2.imshow("low_light_underwater_image.jpg", (degraded_image * 255).astype(np.uint8))
    image_to_save = (degraded_image * 255).astype(np.uint8)  # 转换到 [0, 255] 范围
    cv2.imwrite('degraded_image.png', image_to_save)

    cv2.waitKey(0)
