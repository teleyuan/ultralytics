"""
YOLO 姿态估计模块

该模块提供了 YOLO 姿态估计任务的完整实现,包括:
    - PoseTrainer: 姿态估计训练器
    - PosePredictor: 姿态估计预测器
    - PoseValidator: 姿态估计验证器

主要功能:
    - 训练: 联合训练边界框、关键点和可见性
    - 预测: 检测人体并预测关键点位置
    - 验证: mAP (box), mAP (pose), OKS 评估

损失函数:
    - box_loss: 边界框回归损失
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失
    - kpt_loss: 关键点损失 (OKS Loss)

典型应用:
    - 人体姿态估计
    - 动作识别
    - 运动分析
    - 健身指导
"""

from .predict import PosePredictor
from .train import PoseTrainer
from .val import PoseValidator

__all__ = "PosePredictor", "PoseTrainer", "PoseValidator"
