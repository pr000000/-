# 导入必要模块
import cv2  # OpenCV 图像处理库
import numpy as np  # 数值计算库
import os  # 处理文件路径
from tkinter import *  # Tkinter 图形界面库
from tkinter import filedialog, messagebox  # 文件对话框、消息弹窗
from PIL import Image, ImageTk  # 图像格式转换和图像在Tk中显示

# 初始化主窗口
root = Tk()  # 创建Tkinter主窗口
root.title("图像处理工具")  # 设置窗口标题
root.geometry("1200x700")  # 设置窗口大小

# 初始化图像变量（原图和处理后的图像）
original_image = None  # 原始图像
processed_image = None  # 处理后的图像

# 创建图像显示区域（Label）
img_label = Label(root)  # 创建用于显示图像的标签
img_label.pack()  # 添加到主窗口中

def display_image(img):
    """显示图像到Tkinter界面中"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV的BGR图像转换为RGB
    img_pil = Image.fromarray(img_rgb)  # 将NumPy数组转换为PIL图像对象
    img_pil = img_pil.resize((600, 400))  # 缩放图像到适合窗口显示的尺寸
    img_tk = ImageTk.PhotoImage(img_pil)  # 转为可在Tk中使用的图像对象
    img_label.configure(image=img_tk)  # 将图片显示在标签中
    img_label.image = img_tk  # 保持引用，防止图片被回收

def load_image():
    """导入图片并在界面中显示"""
    global original_image, processed_image
    file_path = filedialog.askopenfilename()  # 弹出文件选择对话框
    if file_path:
        try:
            file_path = os.path.abspath(file_path)  # 获取绝对路径
            img_array = np.fromfile(file_path, dtype=np.uint8)  # 读取为原始字节数组
            original_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为OpenCV图像
            if original_image is None:
                messagebox.showwarning("错误", "无法读取图像，请换一张图片试试。")
                return
            processed_image = original_image.copy()  # 创建处理图像的副本
            display_image(processed_image)  # 显示图像
            apply_changes()  # 应用当前所有设置
        except Exception as e:
            messagebox.showerror("加载失败", str(e))  # 异常处理提示

def apply_changes():
    """根据滑动条设置应用图像处理功能"""
    global processed_image
    if original_image is None:
        return  # 没有图片则不执行任何处理

    # 获取用户调节的各项参数
    brightness = brightness_scale.get()      # 亮度：0-100（默认50）
    contrast = contrast_scale.get()          # 对比度：0-100（默认50）
    hue = hue_scale.get()                    # 色调：-90~90
    saturation = saturation_scale.get()      # 饱和度：0-100（默认50）
    scale = scale_scale.get()                # 缩放：10%-200%
    angle = rotate_scale.get()               # 旋转角度：-180°~180°
    denoise = denoise_scale.get()            # 去噪强度：0-20
    sharpen = sharpen_scale.get()            # 锐化程度：0-5

    # 拷贝原图用于处理，转换为float32类型以避免处理中的精度问题
    img = original_image.copy().astype(np.float32)

    # -------- 图像增强：亮度与对比度 --------
    img = img * (contrast / 50) + (brightness - 50)  # 简化线性增强算法
    img = np.clip(img, 0, 255)  # 限制值在[0,255]之间，避免溢出

    # -------- HSV变换：色调和饱和度调节 --------
    img_hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue) % 180  # 调整色调（H通道）
    img_hsv[:, :, 1] *= (saturation / 50)  # 调整饱和度（S通道）
    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # 转回BGR颜色空间

    # -------- 图像缩放处理 --------
    scale_factor = scale / 100.0  # 缩放比例
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # -------- 图像旋转处理 --------
    (h, w) = img.shape[:2]  # 获取图像尺寸
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)  # 构造旋转矩阵
    img = cv2.warpAffine(img, M, (w, h))  # 应用旋转变换

    # -------- 图像去噪处理（高斯模糊） --------
    if denoise > 0:
        k = int(denoise // 2) * 2 + 1  # 转为奇数核大小
        img = cv2.GaussianBlur(img, (k, k), 0)

    # -------- 图像锐化处理（拉普拉斯增强） --------
    if sharpen > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpen, -1],
                           [0, -1, 0]])  # 根据锐化程度调整中心权重
        img = cv2.filter2D(img, -1, kernel)  # 使用卷积核进行锐化

    # 保存处理后的图像并更新界面显示
    processed_image = np.clip(img, 0, 255).astype(np.uint8)
    display_image(processed_image)

def save_image():
    """保存当前处理后的图像为文件"""
    if processed_image is None:
        messagebox.showwarning("提示", "还没有处理过的图像可以保存。")
        return
    file_path = filedialog.asksaveasfilename(  # 弹出保存对话框
        defaultextension=".jpg",
        filetypes=[("JPEG 文件", "*.jpg"), ("PNG 文件", "*.png"), ("所有文件", "*.*")]
    )
    if file_path:
        try:
            ext = os.path.splitext(file_path)[1]  # 获取文件扩展名
            result, encoded_img = cv2.imencode(ext, processed_image)  # 编码图像
            if result:
                encoded_img.tofile(file_path)  # 保存为文件
                messagebox.showinfo("保存成功", f"图像已保存到：\n{file_path}")
            else:
                raise ValueError("图像编码失败")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存图像失败：\n{e}")

def create_slider(label_text, row, from_, to, initial=50):
    """创建一个带标签的滑动条控件"""
    Label(control_frame, text=label_text).grid(row=row, column=0)  # 标签
    scale = Scale(control_frame, from_=from_, to=to, orient=HORIZONTAL, command=lambda x: apply_changes())  # 滑动条
    scale.set(initial)  # 设置初始值
    scale.grid(row=row, column=1)  # 布局
    return scale  # 返回滑动条对象

# -------- 创建控制面板 --------
control_frame = Frame(root)  # 控制区域框架
control_frame.pack()  # 添加到主窗口

# 导入/保存按钮
Button(control_frame, text="导入图片", command=load_image).grid(row=0, column=0, pady=10)
Button(control_frame, text="保存图片", command=save_image).grid(row=0, column=1, padx=10)

# -------- 添加所有滑动条控件 --------
brightness_scale = create_slider("亮度", 1, 0, 100, 50)
contrast_scale = create_slider("对比度", 2, 0, 100, 50)
hue_scale = create_slider("色调", 3, -90, 90, 0)
saturation_scale = create_slider("饱和度", 4, 0, 100, 50)
scale_scale = create_slider("缩放比例", 5, 10, 200, 100)
rotate_scale = create_slider("旋转角度", 6, -180, 180, 0)
denoise_scale = create_slider("去噪强度", 7, 0, 20, 0)
sharpen_scale = create_slider("锐化程度", 8, 0, 5, 0)

# 启动Tkinter主事件循环
root.mainloop()
