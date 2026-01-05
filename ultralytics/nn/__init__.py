"""
Ultralytics 神经网络模块初始化文件

这个模块是 Ultralytics YOLO 神经网络架构的入口点，提供了所有核心模型类和工具函数的导入。
它定义了模型的基类、特定任务的模型类，以及模型解析和加载的工具函数。

主要功能:
    - 导出基础模型类（BaseModel）和特定任务的模型类
    - 提供模型解析和加载工具函数
    - 支持检测、分类、分割等多种任务的模型
    - 提供模型规模和任务类型的推断功能
"""

# 从 tasks 模块导入核心模型类和工具函数
from .tasks import (
    BaseModel,  # 所有 YOLO 模型的基类
    ClassificationModel,  # 图像分类模型
    DetectionModel,  # 目标检测模型
    SegmentationModel,  # 实例分割模型
    guess_model_scale,  # 从模型路径推断模型规模（n/s/m/l/x）
    guess_model_task,  # 从模型架构或路径推断任务类型
    load_checkpoint,  # 加载模型检查点
    parse_model,  # 解析 YAML 配置文件并构建模型
    torch_safe_load,  # 安全加载 PyTorch 模型权重
    yaml_model_load,  # 从 YAML 文件加载模型配置
)

# 定义模块的公共接口（当使用 from ultralytics.nn import * 时导出的内容）
__all__ = (
    "BaseModel",  # 基础模型类
    "ClassificationModel",  # 分类模型类
    "DetectionModel",  # 检测模型类
    "SegmentationModel",  # 分割模型类
    "guess_model_scale",  # 模型规模推断函数
    "guess_model_task",  # 模型任务推断函数
    "load_checkpoint",  # 模型检查点加载函数
    "parse_model",  # 模型解析函数
    "torch_safe_load",  # 安全加载函数
    "yaml_model_load",  # YAML 配置加载函数
)
