# 图像处理基本程序

这是一个使用 OpenCV 和 NumPy 实现的图像处理基本程序。

## 功能特性

- ✅ 读取和保存图像
- ✅ 灰度转换
- ✅ 图像缩放
- ✅ 高斯模糊
- ✅ 边缘检测（Canny算法）
- ✅ 图像旋转
- ✅ 亮度调整
- ✅ 对比度调整

## 安装依赖

```bash
pip install -r requirements.txt
```

或者直接安装：

```bash
pip install opencv-python numpy
```

## 使用方法

1. 将你的图片文件命名为 `input.jpg` 并放在程序同目录下
2. 或者修改 `aaa.py` 中的 `input_image` 变量为你的图片路径
3. 运行程序：

```bash
python aaa.py
```

## 输出文件

程序会生成以下处理后的图像：

- `output_gray.jpg` - 灰度图
- `output_resized.jpg` - 缩放后的图像（50%大小）
- `output_blur.jpg` - 高斯模糊后的图像
- `output_edges.jpg` - 边缘检测结果
- `output_rotated.jpg` - 旋转45度后的图像
- `output_bright.jpg` - 亮度增强后的图像
- `output_contrast.jpg` - 对比度增强后的图像

## 代码说明

程序提供了多个独立的图像处理函数，可以在其他项目中直接使用：

- `load_image()` - 读取图像
- `convert_to_grayscale()` - 灰度转换
- `resize_image()` - 调整大小
- `apply_blur()` - 高斯模糊
- `detect_edges()` - 边缘检测
- `rotate_image()` - 图像旋转
- `adjust_brightness()` - 亮度调整
- `adjust_contrast()` - 对比度调整
- `save_image()` - 保存图像
- `show_image()` - 显示图像
