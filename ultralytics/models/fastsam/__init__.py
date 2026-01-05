"""
FastSAM (Fast Segment Anything Model) 模块初始化文件

FastSAM 是基于 YOLOv8 的快速分割模型，旨在提供与 SAM 相媲美的分割性能，
但速度更快，更适合实时应用。

主要特点:
    - 基于 YOLO 架构的实时分割
    - 比原始 SAM 快 50 倍以上
    - 支持实例分割和语义分割
    - 轻量级设计，适合边缘设备部署
    - 支持文本提示和边界框提示

导出的类:
    FastSAM: FastSAM 模型主类
    FastSAMPredictor: FastSAM 预测器
    FastSAMValidator: FastSAM 验证器
"""

# 导入 FastSAM 核心组件
from .model import FastSAM  # FastSAM 模型类
from .predict import FastSAMPredictor  # 预测器
from .val import FastSAMValidator  # 验证器

# 定义模块的公共接口
__all__ = "FastSAM", "FastSAMPredictor", "FastSAMValidator"
