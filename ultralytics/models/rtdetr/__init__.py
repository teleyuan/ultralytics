"""
RT-DETR (Real-Time Detection Transformer) 模块初始化文件

RT-DETR 是百度开发的实时目标检测 Transformer 模型，在保持 Transformer 优势的同时
实现了实时检测速度，无需 NMS (非极大值抑制) 后处理。

主要特点:
    - 基于 Transformer 的端到端检测
    - 实时推理速度（可与 YOLO 媲美）
    - 无需 NMS 后处理
    - 高精度和高召回率
    - 支持多尺度特征融合
    - 混合编码器设计（CNN + Transformer）

技术创新:
    - IoU-aware Query Selection: 基于 IoU 的查询选择
    - Uncertainty-minimal Query Selection: 最小化不确定性的查询选择
    - Efficient Hybrid Encoder: 高效混合编码器（AIFI + CCFM）
    - 端到端训练，无需锚框

导出的类:
    RTDETR: RT-DETR 模型主类
    RTDETRPredictor: RT-DETR 预测器
    RTDETRValidator: RT-DETR 验证器
"""

# 导入 RT-DETR 核心组件
from .model import RTDETR  # RT-DETR 模型类
from .predict import RTDETRPredictor  # 预测器
from .val import RTDETRValidator  # 验证器

# 定义模块的公共接口
__all__ = "RTDETR", "RTDETRPredictor", "RTDETRValidator"
