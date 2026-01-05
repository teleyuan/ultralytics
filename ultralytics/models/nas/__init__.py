"""
NAS (Neural Architecture Search) 模块初始化文件

NAS 模型是通过神经架构搜索技术自动设计的目标检测模型，
旨在在特定约束（如速度、精度、模型大小）下找到最优的网络架构。

主要特点:
    - 自动化网络架构设计
    - 针对特定硬件和场景优化
    - 平衡精度和效率
    - 支持移动端和边缘设备部署
    - 基于 YOLO 架构的改进

技术特点:
    - AutoML 驱动的架构搜索
    - 多目标优化（精度、速度、大小）
    - 硬件感知的架构设计
    - 高效的推理性能

导出的类:
    NAS: NAS 模型主类
    NASPredictor: NAS 预测器
    NASValidator: NAS 验证器
"""

# 导入 NAS 核心组件
from .model import NAS  # NAS 模型类
from .predict import NASPredictor  # 预测器
from .val import NASValidator  # 验证器

# 定义模块的公共接口
__all__ = "NAS", "NASPredictor", "NASValidator"
