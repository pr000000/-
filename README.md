# “御掌灵眸” -- 基于Yolov5的手势识别系统

本项目是一个基于 [YOLOv5](https://github.com/ultralytics/yolov5) 和 PyQt5 实现的实时手势识别系统，具备图像捕获、手势检测、置信度调节、截图保存等功能，适用于人机交互、智能控制、教育演示等多种场景。同时，还具备图像处理的基本功能实现，如对比度调节，图片的去噪，锐化，缩放等。



## 功能特性
#### 基本功能

基本的图像处理应用。

1.图片的导入和保存 

2.亮度调节

3.对比度调节

4.色调调节

5.比例缩放

6.旋转角度

7.图片去噪

8.图片锐化

#### 场景功能

##### 手势识别功能

- **实时检测**：30FPS+ 摄像头视频流处理（支持USB/IP摄像头）
- **多手势支持**：  
  ✊ 拳头（Fist） | ✋ 张开手掌（OpenPalm） | ✌ 剪刀手（PeaceSign） | 👍 大拇指（ThumbsUp）
- **智能调节**：  
  - 动态置信度阈值（0.1~0.9 可调）
  - 自动保存识别截图（路径：`./captures/`）
- **模型优化**：  
  基于YOLOv5s的轻量化训练模型（AP@0.5=0.92）



## 安装指南

环境要求

- Python 3.7+
- CUDA 11.3（GPU加速推荐）

请确保安装以下依赖库：

```bash
pip install opencv-python PyQt5 torch torchvision
```

项目依赖 YOLOv5 框架，请将其 clone 至项目根目录：

```bash
git clone https://github.com/ultralytics/yolov5.git
```

确保你已下载好训练好的手势识别模型权重，并替换如下路径：

```bash
.\yolov5\runs\train\gesture_yolov5s_clean3\weights\best.pt
```

快速启动

```bash
python gui.py
```

系统启动后将调用摄像头，展示实时视频画面及识别结果。通过下方滑块可动态调整识别置信度。

