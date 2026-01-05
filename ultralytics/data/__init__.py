"""
Ultralytics 数据处理模块初始化文件

该模块负责所有与数据处理相关的功能，包括数据集加载、数据增强、数据加载器构建等。
它提供了多种数据集类型以支持不同的计算机视觉任务。

主要功能:
    - 提供基础数据集类 BaseDataset
    - 支持多种任务的专用数据集: 分类、目标检测、分割、姿态估计等
    - 构建数据加载器用于训练和推理
    - 支持 Grounding 任务（开放词汇目标检测）
    - 支持多模态数据集（图像+文本）
    - 加载各种推理源（图片、视频、流等）

导出的类:
    BaseDataset: 所有数据集的基类
    ClassificationDataset: 图像分类数据集
    GroundingDataset: Grounding 任务数据集（DINO等）
    SemanticDataset: 语义分割数据集
    YOLOConcatDataset: 多个YOLO数据集的拼接
    YOLODataset: YOLO目标检测数据集
    YOLOMultiModalDataset: 多模态YOLO数据集（支持文本提示）

导出的函数:
    build_dataloader: 构建PyTorch数据加载器
    build_grounding: 构建Grounding数据集
    build_yolo_dataset: 构建YOLO数据集
    load_inference_source: 加载推理数据源
"""

# 从 base 模块导入基础数据集类
from .base import BaseDataset

# 从 build 模块导入数据集和数据加载器构建函数
from .build import (
    build_dataloader,  # 构建数据加载器的主要函数
    build_grounding,  # 构建Grounding任务数据集
    build_yolo_dataset,  # 构建YOLO数据集
    load_inference_source,  # 加载推理数据源（图片、视频、摄像头等）
)

# 从 dataset 模块导入各种专用数据集类
from .dataset import (
    ClassificationDataset,  # 图像分类任务数据集
    GroundingDataset,  # 开放词汇目标检测数据集（DINO、GroundingDINO等）
    SemanticDataset,  # 语义分割数据集
    YOLOConcatDataset,  # 连接多个YOLO数据集的包装类
    YOLODataset,  # 标准YOLO目标检测数据集
    YOLOMultiModalDataset,  # 多模态数据集，支持文本提示的YOLO（YOLOWorld等）
)

# 定义模块的公共接口
# 当使用 from ultralytics.data import * 时，只导出以下内容
__all__ = (
    # 数据集类
    "BaseDataset",  # 基础数据集类
    "ClassificationDataset",  # 分类数据集
    "GroundingDataset",  # Grounding数据集
    "SemanticDataset",  # 语义分割数据集
    "YOLOConcatDataset",  # 拼接数据集
    "YOLODataset",  # YOLO数据集
    "YOLOMultiModalDataset",  # 多模态数据集
    # 构建函数
    "build_dataloader",  # 数据加载器构建函数
    "build_grounding",  # Grounding数据集构建函数
    "build_yolo_dataset",  # YOLO数据集构建函数
    "load_inference_source",  # 推理源加载函数
)
