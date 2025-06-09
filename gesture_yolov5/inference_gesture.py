import os
import sys
import torch
import cv2
from pathlib import Path

# ✅ 添加 yolov5 模块路径
YOLOV5_PATH = r"C:\Users\25823\Desktop\基于yolov5模型的手势识别系统\yolov5"
sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# ====== 参数配置 ======
weights_path = r"C:\Users\25823\Desktop\基于yolov5模型的手势识别系统\yolov5\runs\train\gesture_yolov5s_clean3\weights\best.pt"
imgsz = 640
conf_thres = 60  # 初始值（百分比，范围 0-100）
iou_thres = 45
device = select_device('')
class_names = ['Fist', 'OpenPalm', 'PeaceSign', 'ThumbsUp']
colors = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (128, 0, 255)]

ROOT = Path(__file__).resolve().parent
save_dir = ROOT / 'screenshots'
save_dir.mkdir(parents=True, exist_ok=True)

# ====== 模型初始化 ======
model = DetectMultiBackend(weights_path, device=device)
stride = model.stride
model.eval()

print(f"✅ 模型加载完成，设备：{device}，按 ↑↓ 调节阈值，当前：{conf_thres / 100:.2f}")

# ====== 创建窗口与滑块 ======
cv2.namedWindow("手势识别")
cv2.createTrackbar("Confidence", "手势识别", conf_thres, 100, lambda x: None)

# ====== 打开摄像头 ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败")
    sys.exit(1)

print("✅ 手势识别开始，按 'q' 退出，'s' 截图")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取滑块实时值（转为小数）
    conf_thres = cv2.getTrackbarPos("Confidence", "手势识别")
    conf_thres_float = conf_thres / 100

    # ==== 图像预处理 ====
    img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # ==== 模型推理 ====
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres_float, iou_thres=iou_thres / 100)

    # ==== 结果绘制 ====
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls = int(cls)
                label = f"{class_names[cls]} {conf:.2f}"
                color = colors[cls % len(colors)]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示窗口
    cv2.imshow("手势识别", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = save_dir / f'screenshot_{cv2.getTickCount()}.jpg'
        cv2.imwrite(str(filename), frame)
        print(f"🖼️ 截图已保存：{filename}")
    elif key == 82:  # ↑键：增加阈值
        conf_thres = min(100, conf_thres + 5)
        cv2.setTrackbarPos("Confidence", "手势识别", conf_thres)
    elif key == 84:  # ↓键：降低阈值
        conf_thres = max(0, conf_thres - 5)
        cv2.setTrackbarPos("Confidence", "手势识别", conf_thres)

cap.release()
cv2.destroyAllWindows()
