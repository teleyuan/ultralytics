"""
SAM (Segment Anything Model) 模块初始化文件

该模块提供了 Meta AI 开发的 Segment Anything Model (SAM) 系列模型的接口。
SAM 是一个强大的图像分割模型，支持提示式分割，能够处理多种类型的提示输入。

支持的模型版本:
    - SAM: 原始 Segment Anything Model (基于 ViT)
    - SAM2: 改进版本，支持视频分割和动态交互
    - SAM3: 最新版本，增强的语义分割能力

主要功能:
    - 提示式分割（点、框、掩码提示）
    - 零样本分割能力
    - 实时图像和视频分割
    - 语义分割支持（SAM3）
    - 交互式分割接口

导出的类:
    SAM: SAM 模型主类
    Predictor: 原始 SAM 预测器
    SAM2Predictor: SAM2 图像预测器
    SAM2VideoPredictor: SAM2 视频预测器
    SAM2DynamicInteractivePredictor: SAM2 动态交互预测器
    SAM3Predictor: SAM3 图像预测器
    SAM3VideoPredictor: SAM3 视频预测器
    SAM3SemanticPredictor: SAM3 语义分割预测器
    SAM3VideoSemanticPredictor: SAM3 视频语义分割预测器
"""

# 导入 SAM 模型主类
from .model import SAM

# 导入各种预测器类
from .predict import (
    Predictor,  # 基础 SAM 预测器
    SAM2DynamicInteractivePredictor,  # SAM2 动态交互预测器
    SAM2Predictor,  # SAM2 图像预测器
    SAM2VideoPredictor,  # SAM2 视频预测器
    SAM3Predictor,  # SAM3 图像预测器
    SAM3SemanticPredictor,  # SAM3 语义分割预测器
    SAM3VideoPredictor,  # SAM3 视频预测器
    SAM3VideoSemanticPredictor,  # SAM3 视频语义分割预测器
)

# 定义模块的公共接口
__all__ = (
    "SAM",  # SAM 模型主类
    "Predictor",  # 基础预测器
    "SAM2DynamicInteractivePredictor",  # SAM2 动态交互
    "SAM2Predictor",  # SAM2 预测器
    "SAM2VideoPredictor",  # SAM2 视频预测
    "SAM3Predictor",  # SAM3 预测器
    "SAM3SemanticPredictor",  # SAM3 语义分割
    "SAM3VideoPredictor",  # SAM3 视频预测
    "SAM3VideoSemanticPredictor",  # SAM3 视频语义分割
)  # tuple or list of exportable items
