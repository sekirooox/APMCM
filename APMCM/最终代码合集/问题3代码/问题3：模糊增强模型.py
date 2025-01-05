# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def global_histogram_equalization_rgb(image):
#     """
#     对 RGB 图像进行全局直方图均衡化
#     :param image: 输入 RGB 图像
#     :return: 均衡化后的 RGB 图像
#     """
#     # 拆分通道
#     channels = cv2.split(image)
#     equalized_channels = []
    
#     # 对每个通道分别进行直方图均衡化
#     for channel in channels:
#         equalized_channel = cv2.equalizeHist(channel)
#         equalized_channels.append(equalized_channel)
    
#     # 合并通道
#     equalized_image = cv2.merge(equalized_channels)
#     return equalized_image

# if __name__ == "__main__":
#     # 加载 RGB 图像
#     img = cv2.imread("Attachment/Attachment 1/image_276.jpg")  
    
#     # 进行全局直方图均衡化
#     img_equalized = global_histogram_equalization_rgb(img)

#     # 显示结果
#     cv2.imshow("Original Image", img)
#     cv2.imshow("Enhanced Image", img_equalized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
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

if __name__ == "__main__":
    # 加载 RGB 图像
    img = cv2.imread("Attachment/Attachment 1/image_002.png")  # 替换为你的图像路径
    
    # 应用 CLAHE 方法
    clip_limit=1.2 # 与对比度正相关
    tile_grid_size=(32,32)# 越大计算资源越耗费，但是细节更加明显
    img_enhanced = clahe_rgb(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    # 显示原始图像和增强图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Enhanced Image", img_enhanced)

    # 可选保存增强后的图像
    cv2.imwrite("enhanced_image.jpg", img_enhanced)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
