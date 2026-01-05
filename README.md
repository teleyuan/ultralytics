# Ultralytics YOLO

Ultralytics YOLO 是一个先进的计算机视觉框架，支持目标检测、实例分割、图像分类、姿态估计等任务。

官方文档：https://docs.ultralytics.com/

## 安装

```bash
pip install ultralytics
```

要求：Python>=3.8, PyTorch>=1.8

## 快速开始

### CLI 使用

```bash
# 使用预训练模型进行预测
yolo predict model=yolo11n.pt source='image.jpg'
```

### Python 使用

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 训练
model.train(data="coco8.yaml", epochs=100, imgsz=640)

# 验证
metrics = model.val()

# 预测
results = model("image.jpg")
results[0].show()

# 导出
model.export(format="onnx")
```

## 支持的模型

- YOLO11 (最新)
- YOLOv8
- YOLOv5
- 其他 YOLO 系列模型

## 支持的任务

- 目标检测 (Detection)
- 实例分割 (Segmentation)
- 图像分类 (Classification)
- 姿态估计 (Pose Estimation)
- 定向边界框检测 (OBB)

## 相关链接

- GitHub Issues: https://github.com/ultralytics/ultralytics/issues
- Discord 社区: https://discord.com/invite/ultralytics
