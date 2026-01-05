"""
YOLOE 增强型检测和分割模块

该模块提供了 YOLOE 增强型模型的完整实现,支持视觉提示和文本提示,包括:
    - 训练器: 多种训练模式 (标准、从头训练、视觉提示、位置编码)
    - 预测器: 支持视觉提示的检测和分割
    - 验证器: 支持文本提示和视觉提示的验证

主要功能:
    - 文本提示: 使用文本描述作为类别提示
    - 视觉提示: 使用参考图像中的目标作为检测模板
    - 位置编码: 支持位置编码训练和无位置编码训练
    - 检测和分割: 同时支持目标检测和实例分割任务

训练器类型:
    - YOLOETrainer: 标准 YOLOE 检测训练器
    - YOLOETrainerFromScratch: 从头开始训练
    - YOLOEVPTrainer: 视觉提示训练器
    - YOLOEPETrainer: 位置编码训练器
    - YOLOEPEFreeTrainer: 无位置编码训练器
    - YOLOESegTrainer: 标准 YOLOE 分割训练器
    - YOLOESegTrainerFromScratch: 分割从头训练
    - YOLOESegVPTrainer: 视觉提示分割训练器
    - YOLOEPESegTrainer: 位置编码分割训练器

典型应用:
    - 少样本检测
    - 视觉提示检测
    - 灵活类别检测
    - 实例分割
"""

from .predict import YOLOEVPDetectPredictor, YOLOEVPSegPredictor
from .train import YOLOEPEFreeTrainer, YOLOEPETrainer, YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from .train_seg import YOLOEPESegTrainer, YOLOESegTrainer, YOLOESegTrainerFromScratch, YOLOESegVPTrainer
from .val import YOLOEDetectValidator, YOLOESegValidator

__all__ = [
    "YOLOEDetectValidator",
    "YOLOEPEFreeTrainer",
    "YOLOEPESegTrainer",
    "YOLOEPETrainer",
    "YOLOESegTrainer",
    "YOLOESegTrainerFromScratch",
    "YOLOESegVPTrainer",
    "YOLOESegValidator",
    "YOLOETrainer",
    "YOLOETrainerFromScratch",
    "YOLOEVPDetectPredictor",
    "YOLOEVPSegPredictor",
    "YOLOEVPTrainer",
]
