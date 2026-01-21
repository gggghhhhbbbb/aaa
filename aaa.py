"""
图像处理基本程序
功能包括：读取、显示、灰度化、缩放、模糊、边缘检测等
"""

import cv2
import numpy as np
import os


def load_image(image_path):
    """读取图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return img


def convert_to_grayscale(img):
    """转换为灰度图"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img, width=None, height=None, scale=1.0):
    """调整图像大小"""
    if scale != 1.0:
        h, w = img.shape[:2]
        new_width = int(w * scale)
        new_height = int(h * scale)
        return cv2.resize(img, (new_width, new_height))
    elif width and height:
        return cv2.resize(img, (width, height))
    elif width:
        h, w = img.shape[:2]
        ratio = width / w
        height = int(h * ratio)
        return cv2.resize(img, (width, height))
    elif height:
        h, w = img.shape[:2]
        ratio = height / h
        width = int(w * ratio)
        return cv2.resize(img, (width, height))
    return img


def apply_blur(img, kernel_size=5):
    """应用高斯模糊"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def detect_edges(img, threshold1=50, threshold2=150):
    """边缘检测（Canny算法）"""
    if len(img.shape) == 3:
        img = convert_to_grayscale(img)
    return cv2.Canny(img, threshold1, threshold2)


def rotate_image(img, angle):
    """旋转图像"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


def adjust_brightness(img, value):
    """调整亮度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(img, alpha):
    """调整对比度"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def save_image(img, output_path):
    """保存图像"""
    cv2.imwrite(output_path, img)
    print(f"图像已保存到: {output_path}")


def show_image(img, window_name="图像"):
    """显示图像"""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """主函数 - 演示基本图像处理功能"""
    # 请修改为你的图片路径
    input_image = "input.jpg"  # 输入图片路径
    
    try:
        # 1. 读取图像
        print("正在读取图像...")
        img = load_image(input_image)
        print(f"图像尺寸: {img.shape}")
        
        # 2. 转换为灰度图
        print("转换为灰度图...")
        gray = convert_to_grayscale(img)
        save_image(gray, "output_gray.jpg")
        
        # 3. 缩放图像（缩小到50%）
        print("缩放图像...")
        resized = resize_image(img, scale=0.5)
        save_image(resized, "output_resized.jpg")
        
        # 4. 应用模糊
        print("应用高斯模糊...")
        blurred = apply_blur(img, kernel_size=15)
        save_image(blurred, "output_blur.jpg")
        
        # 5. 边缘检测
        print("边缘检测...")
        edges = detect_edges(img)
        save_image(edges, "output_edges.jpg")
        
        # 6. 旋转图像（45度）
        print("旋转图像...")
        rotated = rotate_image(img, 45)
        save_image(rotated, "output_rotated.jpg")
        
        # 7. 调整亮度
        print("调整亮度...")
        brightened = adjust_brightness(img, 50)
        save_image(brightened, "output_bright.jpg")
        
        # 8. 调整对比度
        print("调整对比度...")
        contrasted = adjust_contrast(img, alpha=1.5)
        save_image(contrasted, "output_contrast.jpg")
        
        print("\n所有处理完成！")
        print("输出文件:")
        print("  - output_gray.jpg (灰度图)")
        print("  - output_resized.jpg (缩放图)")
        print("  - output_blur.jpg (模糊图)")
        print("  - output_edges.jpg (边缘检测)")
        print("  - output_rotated.jpg (旋转图)")
        print("  - output_bright.jpg (亮度调整)")
        print("  - output_contrast.jpg (对比度调整)")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 图片文件 'input.jpg' 存在于当前目录")
        print("2. 或者修改代码中的 input_image 变量为正确的图片路径")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
