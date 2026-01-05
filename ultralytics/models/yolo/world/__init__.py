"""
YOLO-World 开放词汇目标检测模块

该模块提供了 YOLO-World 开放词汇目标检测的训练功能,包括:
    - WorldTrainer: 开放词汇检测训练器

主要功能:
    - 训练: 支持文本提示和视觉特征联合训练
    - 开放词汇检测: 无需预定义类别,支持任意文本描述
    - 零样本检测: 检测训练集中未见过的类别

损失函数:
    - box_loss: 边界框回归损失
    - cls_loss: 分类损失 (与文本嵌入对齐)
    - dfl_loss: 分布焦点损失

典型应用:
    - 开放域目标检测
    - 零样本检测
    - 少样本检测
    - 灵活类别检测
"""

from .train import WorldTrainer

__all__ = ["WorldTrainer"]
