"""
YOLO 有向边界框检测模块 (OBB - Oriented Bounding Box)

该模块提供了 YOLO 有向边界框检测任务的完整实现,包括:
    - OBBTrainer: 有向边界框训练器
    - OBBPredictor: 有向边界框预测器
    - OBBValidator: 有向边界框验证器

主要功能:
    - 训练: 支持旋转角度预测
    - 预测: 生成带旋转角度的边界框 (x, y, w, h, angle)
    - 验证: Rotated mAP 评估

损失函数:
    - box_loss: 有向边界框回归损失 (Probiou Loss)
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失

典型应用:
    - 遥感图像目标检测
    - 文本检测
    - 任意方向目标检测
    - 密集场景检测
"""

from .predict import OBBPredictor
from .train import OBBTrainer
from .val import OBBValidator

__all__ = "OBBPredictor", "OBBTrainer", "OBBValidator"
