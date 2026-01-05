"""
Ultralytics 模型模块初始化文件

该模块提供了 Ultralytics 库支持的所有深度学习模型的统一接口。
包括目标检测、实例分割、图像分割、姿态估计等多种计算机视觉任务的模型。

支持的模型:
    - YOLO: You Only Look Once 系列目标检测模型
    - YOLOE: YOLO Efficient 高效目标检测模型
    - YOLOWorld: 开放词汇目标检测模型（支持文本提示）
    - SAM: Segment Anything Model 通用分割模型（Meta AI）
    - FastSAM: 快速分割模型（基于 YOLOv8）
    - RTDETR: Real-Time DEtection TRansformer 实时检测 Transformer（百度）
    - NAS: Neural Architecture Search 神经架构搜索模型

主要功能:
    - 提供统一的模型加载接口
    - 支持多种预训练权重
    - 支持训练、推理、验证和导出
    - 跨平台兼容（CPU、GPU、移动端）
"""

# 导入各个模型类
from .fastsam import FastSAM  # FastSAM: 快速分割模型
from .nas import NAS  # NAS: 神经架构搜索模型
from .rtdetr import RTDETR  # RT-DETR: 实时检测 Transformer
from .sam import SAM  # SAM: 通用分割模型
from .yolo import YOLO, YOLOE, YOLOWorld  # YOLO 系列模型

# 定义模块的公共接口
__all__ = "NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld"  # allow simpler import
