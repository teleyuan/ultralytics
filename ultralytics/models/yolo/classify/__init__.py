"""
YOLO 图像分类模块

该模块提供了 YOLO 图像分类任务的完整实现,包括:
    - ClassificationTrainer: 图像分类训练器
    - ClassificationPredictor: 图像分类预测器
    - ClassificationValidator: 图像分类验证器

主要功能:
    - 训练: 支持迁移学习、数据增强、学习率调度
    - 预测: Top-K 分类、概率输出、批量推理
    - 验证: Top-1/Top-5 准确率、混淆矩阵

损失函数:
    - CrossEntropyLoss: 交叉熵损失

典型应用:
    - ImageNet 分类
    - 自定义分类任务
    - 特征提取
"""

from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator

__all__ = "ClassificationPredictor", "ClassificationTrainer", "ClassificationValidator"
