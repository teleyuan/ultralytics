"""
YOLO 目标检测模块

该模块提供了 YOLO 目标检测任务的完整实现,包括:
    - DetectionTrainer: 目标检测训练器,用于训练检测模型
    - DetectionPredictor: 目标检测预测器,用于推理预测
    - DetectionValidator: 目标检测验证器,用于模型评估

主要功能:
    - 训练: 支持多尺度训练、数据增强、分布式训练
    - 预测: NMS 后处理、结果可视化、批量推理
    - 验证: mAP 计算、精度召回率曲线、混淆矩阵

损失函数:
    - box_loss: 边界框回归损失 (CIoU Loss)
    - cls_loss: 分类损失 (BCE Loss)
    - dfl_loss: 分布焦点损失 (Distribution Focal Loss)

典型应用:
    - 通用目标检测
    - 自定义数据集训练
    - 实时检测应用
"""

# 导入目标检测相关类
from .predict import DetectionPredictor  # 目标检测预测器
from .train import DetectionTrainer  # 目标检测训练器
from .val import DetectionValidator  # 目标检测验证器

# 定义公开的API
__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
