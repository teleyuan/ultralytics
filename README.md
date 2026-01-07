# Ultralytics YOLO 项目完整指南

Ultralytics YOLO 是一个先进的计算机视觉框架，支持目标检测、实例分割、图像分类、姿态估计等任务。

**官方文档**: https://docs.ultralytics.com/
**GitHub**: https://github.com/ultralytics/ultralytics
**版本**: 8.3.247

---

## 📋 目录

1. [快速开始](#快速开始)
2. [项目结构详解](#项目结构详解)
3. [核心功能模块](#核心功能模块)
4. [使用指南](#使用指南)
5. [高级功能](#高级功能)
6. [扩展与集成](#扩展与集成)
7. [开发指南](#开发指南)

---

## 🚀 快速开始

### 系统要求

- **Python**: >=3.8
- **PyTorch**: >=1.8
- **操作系统**: Windows, Linux, macOS

### 方式一：直接使用本地源码（推荐用于开发）

**这是最直接的方式，无需安装，直接使用源码！**

#### 1. 安装依赖

首先确保安装了必要的依赖包：

```bash
cd c:\Users\yuan1.wang\Desktop\yolo\ultralytics

# 安装核心依赖
pip install torch torchvision  # PyTorch（如果还没安装）
pip install -r requirements.txt

# 可选：安装额外功能的依赖
pip install opencv-python pillow pyyaml requests scipy psutil
```

#### 2. 在源码目录下创建 Python 脚本

**最简单的方式**：直接在源码根目录（`c:\Users\yuan1.wang\Desktop\yolo\ultralytics\`）下创建 Python 文件，无需任何路径配置！

创建文件 `test_yolo.py`：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 图像推理
results = model("test_image.jpg")
results[0].show()
```

运行：
```bash
cd c:\Users\yuan1.wang\Desktop\yolo\ultralytics
python test_yolo.py
```

#### 3. Python API 完整示例

在源码目录下创建 `demo.py`：

```python
from ultralytics import YOLO

# ===== 基础使用 =====

# 1. 加载预训练模型
model = YOLO("yolo11n.pt")

# 2. 图像推理
results = model("test_image.jpg")
results[0].show()  # 显示结果
results[0].save("output.jpg")  # 保存结果

# 3. 批量推理
results = model(["image1.jpg", "image2.jpg", "image3.jpg"])
for i, result in enumerate(results):
    result.save(f"output_{i}.jpg")

# 4. 视频推理
results = model("video.mp4", save=True, show=True)

# 5. 实时摄像头
results = model(source=0, show=True)  # 0 是默认摄像头

# 6. RTSP 流
results = model("rtsp://192.168.1.100:554/stream")

# ===== 训练模型 =====

# 训练自定义模型
model = YOLO("yolo11n.pt")
results = model.train(
    data="data.yaml",      # 数据集配置文件
    epochs=100,            # 训练轮数
    imgsz=640,             # 图像尺寸
    batch=16,              # 批量大小
    device=0,              # GPU 设备（0, 1, 2... 或 'cpu'）
    workers=8,             # 数据加载线程数
    project="runs/train",  # 保存目录
    name="exp"             # 实验名称
)

# ===== 验证模型 =====

model = YOLO("runs/train/exp/weights/best.pt")
metrics = model.val(data="data.yaml")
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")

# ===== 导出模型 =====

model.export(format="onnx")      # 导出为 ONNX
model.export(format="engine")    # 导出为 TensorRT
model.export(format="coreml")    # 导出为 CoreML
model.export(format="tflite")    # 导出为 TFLite

# ===== 目标跟踪 =====

model = YOLO("yolo11n.pt")
results = model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    save=True,
    show=True
)
```

#### 4. 完整项目示例

在源码目录下创建项目结构：

```
ultralytics/              # 源码根目录
├── ultralytics/          # 核心源代码包
├── predict.py            # 检测脚本（你创建的）
├── train.py              # 训练脚本（你创建的）
├── val.py                # 验证脚本（你创建的）
├── data.yaml             # 数据集配置（你创建的）
├── images/               # 测试图像（你创建的）
│   ├── test1.jpg
│   └── test2.jpg
└── dataset/              # 训练数据集（你创建的）
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

**predict.py** - 检测脚本示例：

```python
"""
图像检测脚本
功能：批量处理图像，进行目标检测并保存结果
"""

from ultralytics import YOLO
import os
import cv2
import torch

# ============================================================
# 预测配置参数
# ============================================================

# 模型和路径配置
model_path = "yolov8s.pt"              
image_dir = "../datasets/coco8/images/val"       
output_dir = "outputs/coco8"                 

# 预测参数
conf = 0.25             # 置信度阈值
iou = 0.45              # NMS IoU 阈值
max_det = 300           # 每张图像最大检测数
imgsz = 640             # 图像尺寸
verbose = False         # 是否打印详细信息

# 其他设置
show_info = True        # 是否显示模型信息

# ============================================================

def auto_select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 块 GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 自动使用第一块 GPU
        device = 0
        print(f"\n将使用 GPU 0 进行推理")
        return device
    else:
        print("未检测到 GPU")
        print("将使用 CPU 进行推理")
        return 'cpu'

def main():
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(image_dir):
        print(f"\n错误：图像文件夹 '{image_dir}' 不存在！")
        print(f"请创建该文件夹并放入图像文件")
        return

    print("正在加载模型...")

    device = auto_select_device()
    model = YOLO(model_path)
    #print(model.names)
    #print(model.info())
    #print(model.model)
    model.to(device)
    model.eval()

    if show_info:
        model.info(True, True)

    print(f"\n模型加载完成！类别数: {len(model.names)}")     

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    if not image_files:
        print(f"\n错误：'{image_dir}' 文件夹中没有找到图像文件！")
        return

    print(f"\n找到 {len(image_files)} 张图像，开始处理...\n")

    # 批量处理图像
    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_name)

        print(f"[{idx}/{len(image_files)}] 处理: {img_name}")

        # 推理
        results = model.predict(
            img_path,
            conf = conf,
            iou = iou,
            max_det = max_det,
            imgsz = imgsz,
            verbose = verbose,
            device = device
        )
        result = results[0]

        # 获取检测结果
        boxes = result.boxes
        detections = []

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

                print(f"  - {class_name}: {confidence:.2%} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print(f"  - 未检测到任何目标")

        # 保存结果图像
        output_path = os.path.join(output_dir, img_name)
        result.save(output_path)
        print(f"  结果已保存到: {output_path}\n")

    print(f"完成！所有结果已保存到 '{output_dir}' 文件夹")


if __name__ == "__main__":
    main()
```

**train.py** - 训练脚本示例：

```python
"""
快速训练脚本（使用内置 COCO8 数据集）
功能：使用 YOLO 自带的 coco8 数据集快速测试训练流程
"""

from ultralytics import YOLO
import torch
import os

# ============================================================
# 训练配置参数
# ============================================================

model_path = "yolov8s.pt"
data = "ultralytics/cfg/datasets/coco8.yaml"

# 训练参数
epochs = 2              # 训练轮数
imgsz = 640             # 图像尺寸
batch = 16              # 批量大小
workers = 8             # 数据加载线程数

# 保存设置
project = "runs/train"      # 保存目录
name = "train_coco8"        # 实验名称
exist_ok = False            # 覆盖还是递增实验目录，False 则递增
save = True                 # 保存检查点
save_period = -1            # 仅保存最后和最佳模型

# 训练策略
patience = 50           # EarlyStopping 耐心值
pretrained = True       # 使用预训练权重

# 优化器设置
optimizer = "auto"      # 自动选择优化器
lr0 = 0.01              # 初始学习率
lrf = 0.01              # 最终学习率
momentum = 0.937        # SGD 动量
weight_decay = 0.0005   # 权重衰减
warmup_epochs = 3.0     # 预热轮数

# 损失权重
box = 7.5               # 边界框损失权重
cls = 0.5               # 分类损失权重
dfl = 1.5               # DFL 损失权重

# 数据增强
hsv_h = 0.015           # HSV-Hue 增强
hsv_s = 0.7             # HSV-Saturation 增强
hsv_v = 0.4             # HSV-Value 增强
degrees = 0.0           # 旋转
translate = 0.1         # 平移
scale = 0.5             # 缩放
fliplr = 0.5            # 左右翻转
mosaic = 1.0            # Mosaic 增强

# 其他设置
verbose = True          # 详细输出
amp = True              # 自动混合精度（加速训练）
plots = True            # 生成训练图表

# ============================================================

def auto_select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 块 GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 自动使用第一块 GPU
        device = 0
        print(f"\n将使用 GPU 0 进行训练")
        return device
    else:
        print("未检测到 GPU")
        print("将使用 CPU 进行训练")
        return 'cpu'

def main():
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(data):
        print(f"\n错误：数据集配置文件 '{data}' 不存在！")
        return

    device = auto_select_device()

    # 加载模型
    model = YOLO(model_path, task="detect", verbose=False)
    model.to(device)

    try:
        results = model.train(
            # 数据配置
            data = data,

            # 训练参数
            epochs = epochs,
            imgsz = imgsz,
            batch = batch,
            device = device,
            workers = workers,

            # 保存设置
            project = project,
            name = name,
            exist_ok = exist_ok,
            save = save,
            save_period = save_period,

            # 训练策略
            patience = patience,
            pretrained = pretrained,

            # 优化器设置
            optimizer = optimizer,
            lr0 = lr0,
            lrf = lrf,
            momentum = momentum,
            weight_decay = weight_decay,
            warmup_epochs = warmup_epochs,

            # 损失权重
            box = box,
            cls = cls,
            dfl = dfl,

            # 数据增强
            hsv_h = hsv_h,
            hsv_s = hsv_s,
            hsv_v = hsv_v,
            degrees = degrees,
            translate = translate,
            scale = scale,
            fliplr = fliplr,
            mosaic = mosaic,

            # 其他设置
            verbose = verbose,
            amp = amp,
            plots = plots,
        )

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)

        # 获取训练结果目录
        save_dir = results.save_dir if hasattr(results, 'save_dir') else "runs/train/train_coco8"

        # 验证最佳模型
        print("\n正在验证最佳模型...")
        best_model_path = os.path.join(save_dir, "weights/best.pt")

        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data="coco8.yaml")

            print("\n最佳模型性能指标:")
            print("-" * 60)
            print(f"mAP50-95:  {metrics.box.map:.4f}   (主要指标)")
            print(f"mAP50:     {metrics.box.map50:.4f}  (IoU=0.5 时的 mAP)")
            print(f"mAP75:     {metrics.box.map75:.4f}  (IoU=0.75 时的 mAP)")
            print(f"Precision: {metrics.box.mp:.4f}   (精确率)")
            print(f"Recall:    {metrics.box.mr:.4f}   (召回率)")
            print("-" * 60)
        else:
            print(f"\n警告：未找到最佳模型文件")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

**val.py** - 验证脚本示例：

```python
"""
模型性能测试脚本
功能：测试模型的准确率、召回率、mAP 等指标
"""

from ultralytics import YOLO
import os
import torch

# ============================================================
# 验证配置参数
# ============================================================

# 模型和数据配置
model_path = "yolov8s.pt"                             
data = "ultralytics/cfg/datasets/coco8.yaml"            

# 验证参数
split = 'val'           # 数据集划分: 'val', 'test', 'train'
imgsz = 640             # 图像尺寸
batch = 16              # 批量大小
conf = 0.001            # 置信度阈值（用于计算指标）
iou = 0.6               # NMS IoU 阈值
max_det = 300           # 每张图像最大检测数
workers = 8             # 数据加载线程数

# 保存设置
save_json = False       # 保存为 COCO JSON 格式
save_hybrid = False     # 保存混合标签
verbose = True          # 打印详细信息
plots = True            # 保存图表
project = "runs/val"    # 保存目录
name = "exp"            # 实验名称

# ============================================================

def auto_select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 块 GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 自动使用第一块 GPU
        device = 0
        print(f"\n将使用 GPU 0 进行训练")
        return device
    else:
        print("未检测到 GPU")
        print("将使用 CPU 进行训练")
        return 'cpu'

def main():
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(data):
        print(f"\n错误：数据集配置文件 '{data}' 不存在！")
        return

    device = auto_select_device()
    model = YOLO(model_path)
    model.to(device)

    # 运行验证
    print("\n开始验证...\n")
    print("-" * 60)

    metrics = model.val(
        data = data,
        split = split,
        imgsz = imgsz,
        batch = batch,
        conf = conf,
        iou = iou,
        max_det = max_det,
        device = device,
        workers = workers,
        save_json = save_json,
        save_hybrid = save_hybrid,
        verbose = verbose,
        plots = plots,
        project = project,
        name = name,
    )

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

    # 打印详细指标
    print("\n检测指标 (Detection Metrics):")
    print("-" * 60)

    # mAP 指标
    print("\n1. mAP (Mean Average Precision) - 平均精度均值:")
    print(f"   mAP50-95:  {metrics.box.map:.4f}   主要指标（COCO 标准）")
    print(f"   mAP50:     {metrics.box.map50:.4f}  (IoU=0.5 时的 mAP)")
    print(f"   mAP75:     {metrics.box.map75:.4f}  (IoU=0.75 时的 mAP)")

    # Precision 和 Recall
    print("\n2. Precision (精确率) 和 Recall (召回率):")
    print(f"   Precision: {metrics.box.mp:.4f}   (预测为正的样本中真正为正的比例)")
    print(f"   Recall:    {metrics.box.mr:.4f}   (所有正样本中被正确预测的比例)")

    # F1 Score
    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
        print(f"   F1-Score:  {f1:.4f}   (Precision 和 Recall 的调和平均)")

    # 各类别 mAP
    print("\n3. 各类别 AP (Average Precision):")
    if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap'):
        for idx, ap_value in zip(metrics.box.ap_class_index, metrics.box.ap):
            class_name = model.names[int(idx)]
            print(f"   {class_name:15s}: {ap_value:.4f}")

    print("\n4. 可视化结果:")
    print("   可视化结果已保存到: runs/val/exp/")


if __name__ == "__main__":
    main()
```
#### 5. 如果需要在其他目录调用

如果你的 Python 脚本不在源码目录下，需要添加路径：

```python
import sys
sys.path.insert(0, r"c:\Users\yuan1.wang\Desktop\yolo\ultralytics")

from ultralytics import YOLO
# ... 其他代码
```

---

### 方式二：pip 安装（适合生产环境）

```bash
# 标准安装
pip install ultralytics

# 从源码安装（可编辑模式）
cd c:\Users\yuan1.wang\Desktop\yolo\ultralytics
pip install -e .

# 使用
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model("image.jpg")
```

---

### CLI 命令行使用

如果需要使用 CLI 命令（需要先安装），有两种方式：

**方式 A：直接运行 Python 模块**

```bash
cd c:\Users\yuan1.wang\Desktop\yolo\ultralytics
python -m ultralytics.cfg predict model=yolo11n.pt source=image.jpg
```

**方式 B：安装后使用 yolo 命令**

```bash
pip install -e .
yolo predict model=yolo11n.pt source=image.jpg
yolo train model=yolo11n.pt data=coco8.yaml epochs=100
```

---

## 📁 项目结构详解

### 顶层目录结构

```
ultralytics/
├── ultralytics/          # 核心源代码包
├── examples/             # 社区贡献的示例代码
├── tests/                # 自动化测试套件
├── pyproject.toml        # 项目配置文件
├── requirements.txt      # 核心依赖
└── README.md            # 本文档
```

---

### 🔧 ultralytics/ - 核心源代码包

这是项目的主要源代码目录，包含所有核心功能模块。

#### 1. **cfg/** - 配置管理中心

**作用**: 存储所有配置文件，管理模型架构、数据集、训练参数等。

```
cfg/
├── __init__.py           # CLI 入口点，配置加载与验证
├── default.yaml          # 默认超参数配置
├── datasets/             # 数据集配置文件（36 个）
│   ├── coco.yaml        # COCO 数据集配置
│   ├── coco8.yaml       # COCO8 小型数据集
│   ├── VOC.yaml         # Pascal VOC 配置
│   ├── ImageNet.yaml    # ImageNet 分类数据集
│   └── ...              # 更多数据集配置
├── models/               # 模型架构 YAML 文件（90+ 个）
│   ├── 11/              # YOLO11 系列
│   │   ├── yolo11n.yaml       # nano 版本
│   │   ├── yolo11s.yaml       # small 版本
│   │   ├── yolo11m.yaml       # medium 版本
│   │   ├── yolo11l.yaml       # large 版本
│   │   ├── yolo11x.yaml       # xlarge 版本
│   │   ├── yolo11-seg.yaml    # 分割模型
│   │   ├── yolo11-pose.yaml   # 姿态估计
│   │   ├── yolo11-obb.yaml    # 定向边界框
│   │   └── yolo11-cls.yaml    # 分类模型
│   ├── 12/              # YOLO12 系列（最新）
│   ├── v8/              # YOLOv8 系列
│   ├── v5/              # YOLOv5 系列
│   ├── v3/              # YOLOv3 系列
│   ├── rt-detr/         # RT-DETR 模型
│   └── README.md        # 模型配置文档
└── trackers/             # 目标跟踪器配置
    ├── botsort.yaml     # BoT-SORT 配置
    ├── bytetrack.yaml   # ByteTrack 配置
    └── README.md        # 跟踪文档
```

**使用示例**:
```python
# 加载特定版本模型
model = YOLO("cfg/models/11/yolo11n.yaml")  # 从配置文件创建新模型
model = YOLO("yolo11n.pt")                   # 加载预训练权重

# 使用自定义数据集配置
model.train(data="cfg/datasets/coco8.yaml", epochs=100)
```

---

#### 2. **data/** - 数据处理模块

**作用**: 处理数据加载、预处理、增强、格式转换。

```
data/
├── __init__.py          # 数据模块初始化
├── base.py              # 基础数据集类
├── build.py             # DataLoader 构建器
├── dataset.py           # 数据集实现（检测、分割、分类等）
├── augment.py           # 数据增强（130KB，核心增强函数）
├── loaders.py           # 多种数据加载器（图片、视频、流媒体）
├── converter.py         # 数据集格式转换（COCO、YOLO、VOC 等）
├── annotator.py         # 自动标注工具
├── split.py             # 数据集划分工具
├── split_dota.py        # DOTA 数据集专用划分
├── utils.py             # 数据工具函数
└── scripts/             # 数据集下载脚本
    ├── download_weights.sh   # 下载预训练权重
    ├── get_coco.sh           # 下载 COCO 数据集
    ├── get_coco128.sh        # 下载 COCO128 数据集
    └── get_imagenet.sh       # 下载 ImageNet
```

**核心功能**:

1. **数据增强** ([augment.py](ultralytics/data/augment.py)):
   - Mosaic、MixUp、CopyPaste
   - 随机翻转、旋转、缩放
   - 颜色抖动、HSV 变换
   - Albumentations 集成

2. **数据加载器** ([loaders.py](ultralytics/data/loaders.py)):
   - 图片加载器
   - 视频加载器
   - RTSP/RTMP 流加载器
   - 屏幕截图加载器
   - YouTube 视频加载器

3. **格式转换** ([converter.py](ultralytics/data/converter.py)):
   ```python
   from ultralytics.data.converter import convert_coco

   # COCO 转 YOLO 格式
   convert_coco(labels_dir='coco/annotations/')
   ```

**使用示例**:
```python
from ultralytics.data import build_dataloader
from ultralytics.data.augment import Albumentations

# 自定义数据加载
dataloader = build_dataloader(
    dataset_path="path/to/dataset",
    batch_size=16,
    workers=8,
    augment=True
)
```

---

#### 3. **engine/** - 核心引擎

**作用**: 实现训练、验证、预测、导出的核心流程。

```
engine/
├── __init__.py          # 引擎模块初始化
├── model.py             # 模型基类（52KB）
├── trainer.py           # 训练管道（47KB）
├── validator.py         # 验证管道（18KB）
├── predictor.py         # 预测管道（24KB）
├── results.py           # 结果处理和可视化（66KB）
├── exporter.py          # 模型导出器（75KB，支持 15+ 格式）
└── tuner.py             # 超参数调优（24KB）
```

**核心类**:

1. **Model** ([model.py](ultralytics/engine/model.py)):
   ```python
   from ultralytics import YOLO

   model = YOLO("yolo11n.pt")

   # 支持的方法
   model.train(data="coco8.yaml", epochs=100)
   model.val()
   model.predict(source="image.jpg")
   model.export(format="onnx")
   model.track(source="video.mp4")
   model.benchmark()
   ```

2. **Trainer** ([trainer.py](ultralytics/engine/trainer.py)):
   - 支持单 GPU 和多 GPU 训练
   - 自动混合精度（AMP）
   - 梯度累积
   - EarlyStopping
   - 模型检查点保存

3. **Exporter** ([exporter.py](ultralytics/engine/exporter.py)):
   ```python
   # 支持的导出格式
   formats = [
       "torchscript",  # TorchScript
       "onnx",         # ONNX
       "openvino",     # OpenVINO
       "engine",       # TensorRT
       "coreml",       # CoreML
       "saved_model",  # TensorFlow SavedModel
       "pb",           # TensorFlow GraphDef
       "tflite",       # TensorFlow Lite
       "edgetpu",      # TensorFlow Edge TPU
       "tfjs",         # TensorFlow.js
       "paddle",       # PaddlePaddle
       "ncnn",         # NCNN
       "mlmodel",      # CoreML
   ]

   model.export(format="onnx", dynamic=True, simplify=True)
   ```

---

#### 4. **models/** - 模型架构实现

**作用**: 实现所有支持的模型架构。

```
models/
├── __init__.py          # 导出所有模型类
├── yolo/                # YOLO 系列模型
│   ├── model.py         # YOLO 模型包装器
│   ├── detect/          # 目标检测
│   │   ├── train.py     # 检测训练器
│   │   ├── val.py       # 检测验证器
│   │   └── predict.py   # 检测预测器
│   ├── segment/         # 实例分割
│   ├── classify/        # 图像分类
│   ├── pose/            # 姿态估计
│   ├── obb/             # 定向边界框检测
│   ├── world/           # 开放词汇检测（YOLOWorld）
│   └── yoloe/           # 高效 YOLO 变体
├── sam/                 # Segment Anything Model
│   ├── model.py         # SAM 模型
│   ├── predict.py       # SAM 预测
│   ├── amg.py           # 自动掩码生成
│   ├── modules/         # SAM 组件
│   └── sam3/            # SAM 3 实现
├── fastsam/             # Fast SAM
├── rtdetr/              # Real-Time DETR
│   ├── model.py         # RT-DETR 模型
│   ├── train.py         # RT-DETR 训练
│   ├── val.py           # RT-DETR 验证
│   └── predict.py       # RT-DETR 预测
├── nas/                 # 神经架构搜索模型
└── utils/               # 模型工具
    ├── loss.py          # 损失函数
    └── ops.py           # 操作函数
```

**支持的任务**:

1. **目标检测 (detect)**:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   results = model("image.jpg")
   ```

2. **实例分割 (segment)**:
   ```python
   model = YOLO("yolo11n-seg.pt")
   results = model("image.jpg")
   masks = results[0].masks  # 获取分割掩码
   ```

3. **姿态估计 (pose)**:
   ```python
   model = YOLO("yolo11n-pose.pt")
   results = model("people.jpg")
   keypoints = results[0].keypoints  # 获取关键点
   ```

4. **图像分类 (classify)**:
   ```python
   model = YOLO("yolo11n-cls.pt")
   results = model("image.jpg")
   ```

5. **定向边界框 (obb)**:
   ```python
   model = YOLO("yolo11n-obb.pt")
   results = model("aerial_image.jpg")
   ```

---

#### 5. **nn/** - 神经网络模块

**作用**: 提供神经网络的基础构建块。

```
nn/
├── __init__.py          # NN 模块初始化
├── autobackend.py       # 自动推理后端（支持多种格式）
├── tasks.py             # 任务特定模型定义（71KB）
├── text_model.py        # 文本模型（用于开放词汇检测）
└── modules/             # 可重用神经网络组件
    ├── __init__.py
    ├── activation.py    # 激活函数
    ├── block.py         # 构建块（C2f、C3、Bottleneck 等）
    ├── conv.py          # 卷积层
    ├── head.py          # 检测/分割头
    ├── transformer.py   # Transformer 模块
    └── utils.py         # NN 工具
```

**核心组件**:

- **C2f**: CSPNet 变体，用于特征提取
- **SPPF**: 空间金字塔池化
- **Detect Head**: 检测头
- **Segment Head**: 分割头
- **Pose Head**: 姿态估计头

---

#### 6. **trackers/** - 目标跟踪

**作用**: 实现多目标跟踪算法。

```
trackers/
├── __init__.py          # 跟踪器导出
├── basetrack.py         # 基础跟踪类
├── bot_sort.py          # BoT-SORT 实现
├── byte_tracker.py      # ByteTrack 实现
├── track.py             # 跟踪接口
├── utils/               # 跟踪工具
└── README.md            # 跟踪文档
```

**使用示例**:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 使用 ByteTrack
results = model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    save=True
)

# 使用 BoT-SORT
results = model.track(
    source="video.mp4",
    tracker="botsort.yaml",
    save=True
)
```

---

#### 7. **solutions/** - 即用型应用

**作用**: 提供预构建的计算机视觉解决方案。

```
solutions/
├── __init__.py                    # Solutions 导出
├── solutions.py                   # 基础解决方案框架（41KB）
├── config.py                      # 解决方案配置
├── object_counter.py              # 对象计数
├── object_cropper.py              # 对象裁剪
├── object_blurrer.py              # 对象模糊
├── heatmap.py                     # 热力图生成
├── speed_estimation.py            # 速度估计
├── distance_calculation.py        # 距离计算
├── queue_management.py            # 队列管理
├── region_counter.py              # 区域计数
├── ai_gym.py                      # 健身追踪
├── vision_eye.py                  # 视觉眼追踪
├── instance_segmentation.py       # 实例分割
├── parking_management.py          # 停车管理
├── security_alarm.py              # 安全警报
├── analytics.py                   # 分析图表
├── similarity_search.py           # 视觉相似度搜索
├── streamlit_inference.py         # Streamlit 推理 UI
├── trackzone.py                   # 区域追踪
└── templates/                     # HTML 模板
    └── similarity-search.html     # 相似度搜索 UI
```

**CLI 快捷命令**:
```bash
# 对象计数
yolo solutions count source=video.mp4

# 热力图
yolo solutions heatmap source=video.mp4

# 速度估计
yolo solutions speed source=video.mp4

# 队列管理
yolo solutions queue source=video.mp4

# Streamlit 推理界面
yolo solutions inference
```

**Python 使用**:
```python
from ultralytics.solutions import ObjectCounter

counter = ObjectCounter()
counter.count(source="video.mp4")
```

---

#### 8. **utils/** - 工具函数库

**作用**: 提供共享的工具函数和辅助功能。

```
utils/
├── __init__.py          # 核心工具和常量（64KB）
├── checks.py            # 系统和依赖检查（44KB）
├── downloads.py         # 文件下载工具（23KB）
├── torch_utils.py       # PyTorch 工具（41KB）
├── plotting.py          # 可视化和绘图（49KB）
├── metrics.py           # 评估指标（71KB）
├── loss.py              # 损失函数（43KB）
├── ops.py               # 张量操作（31KB）
├── nms.py               # 非极大值抑制（15KB）
├── tal.py               # 任务对齐学习（21KB）
├── instance.py          # 实例工具（19KB）
├── logger.py            # 日志配置（20KB）
├── autobatch.py         # 自动批量大小（5KB）
├── autodevice.py        # 自动设备选择（9KB）
├── benchmarks.py        # 模型基准测试（35KB）
├── files.py             # 文件操作（8KB）
├── callbacks/           # 训练回调（11 个集成）
│   ├── base.py          # 基础回调类
│   ├── tensorboard.py   # TensorBoard 集成
│   ├── wb.py            # Weights & Biases
│   ├── mlflow.py        # MLflow
│   ├── clearml.py       # ClearML
│   ├── comet.py         # Comet
│   ├── neptune.py       # Neptune
│   ├── raytune.py       # Ray Tune
│   └── ...              # 更多回调
└── export/              # 导出工具
    ├── engine.py        # TensorRT 引擎
    ├── imx.py           # IMX 平台
    └── tensorflow.py    # TensorFlow 助手
```

**常用工具**:

1. **系统检查**:
   ```python
   from ultralytics.utils.checks import check_requirements
   check_requirements(['torch>=1.8.0', 'opencv-python'])
   ```

2. **指标计算**:
   ```python
   from ultralytics.utils.metrics import box_iou, ConfusionMatrix
   ```

3. **可视化**:
   ```python
   from ultralytics.utils.plotting import Annotator
   ```

---

#### 9. **hub/** - Ultralytics HUB 集成

**作用**: 连接到 Ultralytics HUB 云平台，实现云端训练和模型管理。

```
hub/
├── __init__.py          # HUB 初始化
├── auth.py              # 认证处理
├── session.py           # 训练会话管理
├── utils.py             # HUB 工具
└── google/              # Google Colab 集成
```

**使用方法**:
```python
from ultralytics import YOLO, hub

# 登录 HUB
hub.login('your_api_key')

# 从 HUB 加载模型
model = YOLO('https://hub.ultralytics.com/models/xxx')

# 训练并自动上传到 HUB
model.train(data='coco8.yaml', epochs=100)
```

---

### 📚 examples/ - 示例代码

**作用**: 社区贡献的集成示例和教程。

```
examples/
├── README.md                              # 示例概览
├── tutorial.ipynb                         # 入门教程
├── hub.ipynb                              # HUB 集成教程
├── heatmaps.ipynb                         # 热力图教程
├── object_counting.ipynb                  # 对象计数教程
├── object_tracking.ipynb                  # 对象跟踪教程
├── YOLOv8-CPP-Inference/                  # C++ ONNX 推理
├── YOLOv8-ONNXRuntime-CPP/                # C++ ONNXRuntime
├── YOLOv8-LibTorch-CPP-Inference/         # C++ LibTorch
├── YOLOv8-OpenVINO-CPP-Inference/         # C++ OpenVINO
├── YOLOv8-ONNXRuntime-Rust/               # Rust ONNXRuntime
├── YOLOv8-ONNXRuntime/                    # Python ONNXRuntime
├── YOLOv8-OpenCV-ONNX-Python/             # Python OpenCV ONNX
├── YOLOv8-TFLite-Python/                  # Python TFLite
├── YOLOv8-Action-Recognition/             # 动作识别
├── YOLOv8-SAHI-Inference-Video/           # SAHI 切片推理
└── YOLO-Interactive-Tracking-UI/          # 交互式跟踪 UI
```

**查看示例**:
```bash
cd examples
jupyter notebook tutorial.ipynb
```

---

### 🧪 tests/ - 测试套件

**作用**: 自动化测试，确保代码质量。

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # pytest 配置和夹具
├── test_cli.py              # CLI 命令测试
├── test_cuda.py             # CUDA/GPU 测试
├── test_engine.py           # 引擎组件测试
├── test_exports.py          # 导出功能测试
├── test_integrations.py     # 第三方集成测试
├── test_python.py           # Python API 测试
└── test_solutions.py        # Solutions 测试
```

**运行测试**:
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_python.py

# 运行慢速测试
pytest tests/ --slow
```

---

## 🎯 核心功能模块

### 1. 支持的模型架构

| 模型系列 | 任务 | 版本 |
|---------|------|------|
| YOLO11  | 检测、分割、分类、姿态、OBB | n, s, m, l, x |
| YOLO12  | 检测、分割、分类、姿态、OBB | n, s, m, l, x |
| YOLOv8  | 检测、分割、分类、姿态、OBB | n, s, m, l, x |
| YOLOv5  | 检测、分割、分类 | n, s, m, l, x |
| RT-DETR | 检测 | l, x |
| SAM     | 分割 | b, l, h |
| FastSAM | 分割 | s, x |
| YOLOWorld | 开放词汇检测 | s, m, l |

### 2. 支持的任务

#### 目标检测 (Detection)
- 标准边界框检测
- 多类别检测
- 小目标检测优化

#### 实例分割 (Segmentation)
- 像素级精确分割
- 多实例分割
- 全景分割

#### 图像分类 (Classification)
- ImageNet 预训练
- 迁移学习
- 多标签分类

#### 姿态估计 (Pose Estimation)
- 人体关键点检测（17 个关键点）
- 多人姿态估计
- 实时姿态跟踪

#### 定向边界框 (OBB)
- 旋转目标检测
- 航拍图像检测
- 文本检测

### 3. 支持的操作模式

| 模式 | 命令 | 说明 |
|------|------|------|
| train | `model.train()` | 训练模型 |
| val | `model.val()` | 验证模型 |
| predict | `model.predict()` | 推理预测 |
| export | `model.export()` | 导出模型 |
| track | `model.track()` | 目标跟踪 |
| benchmark | `model.benchmark()` | 性能基准测试 |

---

## 📖 使用指南

### 1. 训练自定义模型

#### 准备数据集

**YOLO 格式数据集结构**:
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

**标注格式** (labels/xxx.txt):
```
# class_id x_center y_center width height (归一化到 0-1)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

**数据集配置文件** (data.yaml):
```yaml
path: /path/to/dataset  # 数据集根目录
train: images/train     # 训练图像路径（相对于 path）
val: images/val         # 验证图像路径（相对于 path）

# 类别
names:
  0: person
  1: car
  2: dog
```

#### 开始训练

**CLI 训练**:
```bash
yolo train \
  model=yolo11n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/train \
  name=exp
```

**Python 训练**:
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 训练
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/train",
    name="exp",
    patience=50,           # EarlyStopping 耐心值
    save=True,             # 保存检查点
    save_period=10,        # 每 10 个 epoch 保存一次
    cache=True,            # 缓存图像到内存
    workers=8,             # DataLoader 工作线程数
    optimizer="SGD",       # 优化器：SGD, Adam, AdamW
    lr0=0.01,              # 初始学习率
    lrf=0.01,              # 最终学习率（lr0 * lrf）
    momentum=0.937,        # SGD 动量/Adam beta1
    weight_decay=0.0005,   # 权重衰减
    warmup_epochs=3.0,     # 预热 epoch 数
    warmup_momentum=0.8,   # 预热初始动量
    box=7.5,               # 边界框损失权重
    cls=0.5,               # 分类损失权重
    dfl=1.5,               # DFL 损失权重
    hsv_h=0.015,           # HSV-Hue 增强
    hsv_s=0.7,             # HSV-Saturation 增强
    hsv_v=0.4,             # HSV-Value 增强
    degrees=0.0,           # 旋转角度
    translate=0.1,         # 平移
    scale=0.5,             # 缩放
    shear=0.0,             # 剪切
    perspective=0.0,       # 透视变换
    flipud=0.0,            # 上下翻转概率
    fliplr=0.5,            # 左右翻转概率
    mosaic=1.0,            # Mosaic 增强概率
    mixup=0.0,             # MixUp 增强概率
    copy_paste=0.0,        # Copy-Paste 增强概率
)
```

#### 多 GPU 训练

```bash
# 使用 torch.distributed
yolo train model=yolo11n.pt data=data.yaml device=0,1,2,3 batch=64
```

```python
# Python DDP
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="data.yaml", device=[0, 1, 2, 3], batch=64)
```

---

### 2. 模型验证

```bash
# CLI
yolo val model=runs/train/exp/weights/best.pt data=data.yaml
```

```python
# Python
model = YOLO("runs/train/exp/weights/best.pt")
metrics = model.val(data="data.yaml")

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

---

### 3. 模型推理

#### 图像推理

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 单张图像
results = model("image.jpg")
result = results[0]

# 访问结果
boxes = result.boxes          # 边界框
masks = result.masks          # 分割掩码（如果是分割模型）
keypoints = result.keypoints  # 关键点（如果是姿态模型）
probs = result.probs          # 类别概率（如果是分类模型）

# 绘制结果
result.show()                 # 显示图像
result.save("result.jpg")     # 保存图像

# 获取边界框信息
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # 边界框坐标
    conf = box.conf[0]             # 置信度
    cls = box.cls[0]               # 类别
    print(f"Class: {cls}, Conf: {conf:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
```

#### 批量推理

```python
# 批量推理多张图像
results = model(["image1.jpg", "image2.jpg", "image3.jpg"])

for i, result in enumerate(results):
    result.save(f"result_{i}.jpg")
```

#### 视频推理

```python
# 视频推理
results = model("video.mp4", save=True, show=True)

# 流式处理视频（逐帧）
for result in model("video.mp4", stream=True):
    boxes = result.boxes
    # 处理每一帧
```

#### 实时摄像头推理

```python
# 使用摄像头
results = model(source=0, show=True)  # 0 是默认摄像头
```

#### RTSP 流推理

```python
# RTSP 流
results = model("rtsp://192.168.1.100:554/stream", show=True)
```

---

### 4. 模型导出

Ultralytics 支持导出到 15+ 种格式：

| 格式 | 命令 | 平台 |
|------|------|------|
| PyTorch | `format='torchscript'` | 所有平台 |
| ONNX | `format='onnx'` | 所有平台 |
| OpenVINO | `format='openvino'` | Intel CPU/GPU |
| TensorRT | `format='engine'` | NVIDIA GPU |
| CoreML | `format='coreml'` | iOS/macOS |
| TFLite | `format='tflite'` | Android/移动设备 |
| TensorFlow | `format='saved_model'` | TensorFlow 生态 |
| PaddlePaddle | `format='paddle'` | 百度生态 |
| NCNN | `format='ncnn'` | 移动端 |

**导出示例**:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 导出为 ONNX（推荐用于跨平台部署）
model.export(
    format="onnx",
    dynamic=True,      # 动态输入尺寸
    simplify=True,     # 简化模型
    opset=12           # ONNX opset 版本
)

# 导出为 TensorRT（最快的 GPU 推理）
model.export(
    format="engine",
    device=0,
    half=True,         # FP16 精度
    workspace=4        # GPU 内存（GB）
)

# 导出为 TFLite（Android 部署）
model.export(
    format="tflite",
    int8=True,         # INT8 量化
    data="data.yaml"   # 校准数据
)

# 导出为 CoreML（iOS 部署）
model.export(format="coreml")
```

**使用导出的模型**:

```python
# 加载 ONNX 模型
model = YOLO("yolo11n.onnx")
results = model("image.jpg")

# 加载 TensorRT 模型
model = YOLO("yolo11n.engine")
results = model("image.jpg")
```

---

### 5. 目标跟踪

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# ByteTrack 跟踪
results = model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",  # 或 "botsort.yaml"
    show=True,
    save=True,
    conf=0.3,         # 置信度阈值
    iou=0.5,          # IoU 阈值
    persist=True      # 持久化跟踪 ID
)

# 访问跟踪结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        track_id = box.id   # 跟踪 ID
        cls = box.cls       # 类别
        conf = box.conf     # 置信度
```

---

### 6. 超参数调优

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 使用 Ray Tune 自动调优
model.tune(
    data="data.yaml",
    epochs=30,
    iterations=300,      # 调优迭代次数
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False
)
```

---

## 🚀 高级功能

### 1. 自定义数据增强

```python
from ultralytics.data.augment import Albumentations
import albumentations as A

# 自定义 Albumentations 增强
augment = Albumentations(
    transforms=[
        A.Blur(p=0.5),
        A.MedianBlur(p=0.5),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
    ]
)

# 在训练时使用
model.train(data="data.yaml", augment=augment)
```

### 2. 自定义回调

```python
from ultralytics import YOLO
from ultralytics.utils.callbacks import add_integration_callbacks

def on_train_start(trainer):
    print("Training started!")

def on_epoch_end(trainer):
    print(f"Epoch {trainer.epoch} finished")

# 添加回调
model = YOLO("yolo11n.pt")
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_epoch_end", on_epoch_end)

model.train(data="data.yaml", epochs=10)
```

### 3. Weights & Biases 集成

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# W&B 会自动检测并集成
model.train(
    data="data.yaml",
    epochs=100,
    project="my-project",  # W&B 项目名
    name="yolo11n-run"     # W&B 运行名
)
```

### 4. TensorBoard 集成

```python
# TensorBoard 自动启用
model.train(data="data.yaml", epochs=100)

# 查看 TensorBoard
# tensorboard --logdir runs/train
```

### 5. 模型集成（Ensemble）

```python
from ultralytics import YOLO

# 加载多个模型
models = [
    YOLO("yolo11n.pt"),
    YOLO("yolo11s.pt"),
    YOLO("yolo11m.pt")
]

# 集成预测
results = []
for model in models:
    results.append(model("image.jpg"))

# 合并结果（自定义逻辑）
```

### 6. 模型剪枝和量化

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 导出为 INT8 量化模型
model.export(
    format="onnx",
    int8=True,
    data="data.yaml"  # 用于校准的数据
)
```

---

## 🔗 扩展与集成

### 1. 支持的日志和实验追踪

- **Weights & Biases** (wandb)
- **TensorBoard**
- **MLflow**
- **ClearML**
- **Comet**
- **Neptune**
- **Ray Tune**

### 2. 支持的导出框架

- **PyTorch**: TorchScript
- **ONNX**: ONNX Runtime
- **TensorFlow**: SavedModel, TFLite, TFJS
- **OpenVINO**: Intel 推理引擎
- **TensorRT**: NVIDIA 推理引擎
- **CoreML**: Apple 设备
- **PaddlePaddle**: 百度深度学习框架
- **NCNN**: 腾讯移动端框架
- **MNN**: 阿里移动端框架

### 3. 部署方式

#### Docker 部署

```dockerfile
FROM ultralytics/ultralytics:latest

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

```bash
docker run -it --gpus all ultralytics/ultralytics:latest
```

#### REST API 部署

```python
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO("yolo11n.pt")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img)

    return jsonify({
        'boxes': results[0].boxes.xyxy.tolist(),
        'scores': results[0].boxes.conf.tolist(),
        'classes': results[0].boxes.cls.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Streamlit 应用

```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLO Object Detection")

model = YOLO("yolo11n.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model(image)
    st.image(results[0].plot(), caption="Detection Results")
```

---

## 🛠 开发指南

### 1. 从源码安装

```bash
# 克隆仓库
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# 安装开发依赖
pip install -e ".[dev]"

# 安装所有可选依赖
pip install -e ".[dev,export,solutions,logging,extra]"
```

### 2. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_python.py

# 运行带覆盖率的测试
pytest --cov=ultralytics tests/

# 运行慢速测试
pytest tests/ --slow
```

### 3. 代码格式化

```bash
# 使用 ruff 格式化代码
ruff format ultralytics/

# 检查代码风格
ruff check ultralytics/

# 自动修复
ruff check --fix ultralytics/
```

### 4. 贡献代码

1. Fork 仓库
2. 创建功能分支: `git checkout -b feature/my-feature`
3. 提交更改: `git commit -m 'Add my feature'`
4. 推送分支: `git push origin feature/my-feature`
5. 创建 Pull Request

### 5. 自定义模型架构

在 `ultralytics/cfg/models/` 中创建新的 YAML 配置:

```yaml
# my_custom_model.yaml
nc: 80  # 类别数
depth_multiple: 0.33
width_multiple: 0.50

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  # ... 更多层

head:
  - [-1, 1, Detect, [nc]]
```

加载自定义模型:

```python
model = YOLO("my_custom_model.yaml")
model.train(data="data.yaml")
```

---

## 📊 性能基准

### 模型速度对比（COCO val2017）

| 模型 | 尺寸 | mAP50-95 | 速度 CPU (ms) | 速度 T4 (ms) |
|------|------|----------|--------------|--------------|
| YOLO11n | 640 | 39.5 | 56.1 | 1.5 |
| YOLO11s | 640 | 47.0 | 90.0 | 2.5 |
| YOLO11m | 640 | 51.5 | 183.2 | 4.7 |
| YOLO11l | 640 | 53.4 | 238.6 | 6.2 |
| YOLO11x | 640 | 54.7 | 462.8 | 11.3 |

---

## 🔍 常见问题

### 1. CUDA 内存不足

```python
# 减小批量大小
model.train(data="data.yaml", batch=8)

# 使用自动批量大小
model.train(data="data.yaml", batch=-1)

# 使用混合精度训练
model.train(data="data.yaml", amp=True)
```

### 2. 训练速度慢

```python
# 启用缓存
model.train(data="data.yaml", cache=True)

# 增加工作线程
model.train(data="data.yaml", workers=8)

# 使用更小的图像尺寸
model.train(data="data.yaml", imgsz=416)
```

### 3. 检测精度低

- 增加训练 epochs
- 使用更大的模型（n → s → m → l → x）
- 调整数据增强参数
- 检查数据集质量和标注
- 调整置信度阈值

---

## 📞 获取帮助

- **文档**: https://docs.ultralytics.com/
- **GitHub Issues**: https://github.com/ultralytics/ultralytics/issues
- **Discord 社区**: https://discord.com/invite/ultralytics
- **论坛**: https://community.ultralytics.com/

---

## 📄 许可证

Ultralytics YOLO 采用 **AGPL-3.0 许可证**（开源项目）或**企业许可证**（商业应用）。

详见: https://ultralytics.com/license

---

## 🌟 引用

如果您在研究中使用 Ultralytics YOLO，请引用：

```bibtex
@software{ultralytics_yolo,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

**最后更新**: 2026-01-06
**版本**: 8.3.247
