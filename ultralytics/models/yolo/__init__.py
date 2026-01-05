"""
YOLO 模型包

该包提供了 YOLO 系列模型的完整实现,包括多种计算机视觉任务:
    - classify: 图像分类任务
    - detect: 目标检测任务
    - segment: 实例分割任务
    - pose: 姿态估计任务
    - obb: 有向边界框检测任务
    - world: 开放词汇目标检测任务
    - yoloe: 增强型 YOLO 模型 (支持视觉提示)

主要类:
    - YOLO: 标准 YOLO 模型,支持多种任务
    - YOLOWorld: 开放词汇目标检测模型
    - YOLOE: 增强型 YOLO 模型,支持视觉和文本提示

导出的模块和类:
    - YOLO, YOLOE, YOLOWorld: 模型类
    - classify, detect, obb, pose, segment, world, yoloe: 各任务子模块
"""

# 导入各任务子模块
from ultralytics.models.yolo import classify, detect, obb, pose, segment, world, yoloe

# 导入模型类
from .model import YOLO, YOLOE, YOLOWorld

# 定义公开的API
__all__ = "YOLO", "YOLOE", "YOLOWorld", "classify", "detect", "obb", "pose", "segment", "world", "yoloe"
