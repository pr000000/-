import sys, os, time  # 系统相关库，用于路径处理、延时等
import cv2            # OpenCV，用于视频捕捉和图像处理
import torch          # PyTorch，用于深度学习模型加载和推理
from pathlib import Path  # 文件路径处理，更优雅的跨平台支持

# PyQt5 用于构建图形界面
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 添加YOLOv5路径以导入其模块
YOLOV5_PATH = "yolov5"
sys.path.append(YOLOV5_PATH)

# 导入YOLOv5相关模块
from yolov5.models.common import DetectMultiBackend  # 模型加载器
from yolov5.utils.general import non_max_suppression, scale_boxes  # NMS和坐标缩放
from yolov5.utils.augmentations import letterbox  # 图像预处理
from yolov5.utils.torch_utils import select_device  # 设备选择（CPU/GPU）

# 定义主窗口类
class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别系统")  # 设置窗口标题
        self.resize(1100, 600)             # 设置窗口大小

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # 显示识别结果的标签
        self.result_label = QLabel("识别结果：")
        self.result_label.setFixedSize(300, 480)
        self.result_label.setStyleSheet("background-color: white; font-size: 20px; padding: 10px")

        # 创建滑动条用于调整置信度阈值
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.slider_changed)

        self.slider_label = QLabel("缓冲值: 0.20")  # 显示滑动条数值

        # 截图按钮
        self.screenshot_button = QPushButton('📸 截图')
        self.screenshot_button.clicked.connect(self.save_screenshot)

        # 控制区域布局（滑动条 + 截图按钮）
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.screenshot_button)

        # 总体布局
        layout = QVBoxLayout()
        video_result_layout = QHBoxLayout()
        video_result_layout.addWidget(self.video_label)
        video_result_layout.addWidget(self.result_label)

        layout.addLayout(video_result_layout)
        layout.addLayout(control_layout)
        self.setLayout(layout)

        # 创建截图保存目录
        self.save_dir = Path("screenshots")
        self.save_dir.mkdir(exist_ok=True)

        # 初始化YOLO模型
        self.device = select_device('')  # 自动选择设备
        weights = r".\yolov5\runs\train\gesture_yolov5s_clean3\weights\best.pt"  # 权重路径
        self.model = DetectMultiBackend(weights, device=self.device)  # 加载模型
        self.model.eval()
        self.stride = self.model.stride
        self.imgsz = 640  # 输入图像大小
        self.class_names = ['Fist', 'OpenPalm', 'PeaceSign', 'ThumbsUp']  # 类别名
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (128, 0, 255)]  # 每类对应颜色
        self.conf_thres = 0.5  # 初始置信度阈值

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()  # 创建计时器
        self.timer.timeout.connect(self.update_frame)  # 每次超时执行 update_frame
        self.timer.start(30)  # 每30ms更新一次视频帧

    def slider_changed(self, value):
        """更新置信度阈值"""
        self.conf_thres = value / 100
        self.slider_label.setText(f"缓冲值: {self.conf_thres:.2f}")

    def update_frame(self):
        """更新并处理每一帧图像"""
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame = frame.copy()  # 保留原始帧截图用

        # 图像预处理（填充、转RGB）
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB，并转为CHW格式
        img = torch.from_numpy(img.copy()).to(self.device).float() / 255.0
        img = img.unsqueeze(0)  # 添加batch维度

        # 模型推理与NMS
        with torch.no_grad():
            pred = self.model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=0.45)

        result_text = "识别结果：\n"
        for det in pred:
            if len(det):
                # 坐标映射回原图尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    cls = int(cls)
                    label = f"{self.class_names[cls]} {conf:.2f}"
                    color = self.colors[cls % len(self.colors)]
                    # 绘制矩形框与标签
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    result_text += label + "\n"

        # 更新识别结果显示
        self.result_label.setText(result_text)

        # 将OpenCV图像显示到PyQt界面
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def save_screenshot(self):
        """保存当前帧为图片"""
        save_path = self.save_dir / f'screenshot_{int(time.time())}.jpg'
        cv2.imwrite(str(save_path), self.frame)
        print(f"✅ 截图保存成功: {save_path}")

    def closeEvent(self, event):
        """关闭窗口时释放摄像头资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

# 启动应用程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GestureApp()
    win.show()
    sys.exit(app.exec_())
