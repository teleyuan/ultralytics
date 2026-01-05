"""
YOLO 实例分割模块

该模块提供了 YOLO 实例分割任务的完整实现,包括:
    - SegmentationTrainer: 实例分割训练器
    - SegmentationPredictor: 实例分割预测器
    - SegmentationValidator: 实例分割验证器

主要功能:
    - 训练: 支持边界框和掩码联合训练
    - 预测: 生成像素级分割掩码和边界框
    - 验证: mAP (box), mAP (mask) 评估

损失函数:
    - box_loss: 边界框回归损失
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失
    - mask_loss: 掩码损失 (BCE + Dice)

典型应用:
    - 实例分割
    - 语义分割
    - 全景分割
"""

from .predict import SegmentationPredictor
from .train import SegmentationTrainer
from .val import SegmentationValidator

__all__ = "SegmentationPredictor", "SegmentationTrainer", "SegmentationValidator"
