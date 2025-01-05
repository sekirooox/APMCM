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

if __name__ == "__main__":
    img = cv2.imread("Attachment/Attachment 2/test_011.png") 
    cv2.imshow("Original Image", img)
    adjusted_img = gray_world_algorithm(img)

    cv2.imwrite("enhanced_image.jpg", adjusted_img)

    cv2.imshow("Adjusted Image", adjusted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
