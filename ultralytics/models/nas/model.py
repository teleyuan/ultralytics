"""
YOLO-NAS 模型接口模块

YOLO-NAS 是通过神经架构搜索（NAS）技术自动设计的目标检测模型。
它利用 AutoML 技术找到最优的网络架构，在速度和精度之间取得了良好的平衡。

主要特点:
    - AutoML 驱动的架构设计
    - 针对目标检测任务优化
    - 多种模型尺寸（Small、Medium、Large）
    - 基于 Super-Gradients 框架
    - 仅支持预训练模型

技术特点:
    - Neural Architecture Search 自动架构搜索
    - 硬件感知的优化
    - 高效的推理性能
    - 在 COCO 数据集上达到 SOTA 性能
"""

from __future__ import annotations  # 启用延迟类型注解评估

from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

import torch  # PyTorch 深度学习框架

# 导入必要的工具和类
from ultralytics.engine.model import Model  # 基础模型类
from ultralytics.utils import DEFAULT_CFG_DICT  # 默认配置字典
from ultralytics.utils.downloads import attempt_download_asset  # 下载资源
from ultralytics.utils.patches import torch_load  # 安全加载模型
from ultralytics.utils.torch_utils import model_info  # 模型信息工具

# 导入 NAS 组件
from .predict import NASPredictor  # NAS 预测器
from .val import NASValidator  # NAS 验证器


class NAS(Model):
    """YOLO-NAS 目标检测模型。

    该类为 YOLO-NAS 模型提供了统一接口，并扩展了 Ultralytics 引擎的 Model 类。
    它旨在使用预训练或自定义训练的 YOLO-NAS 模型简化目标检测任务。

    属性:
        model (torch.nn.Module): 加载的 YOLO-NAS 模型
        task (str): 模型的任务类型，默认为 'detect'
        predictor (NASPredictor): 用于执行预测的预测器实例
        validator (NASValidator): 用于模型验证的验证器实例

    方法:
        info: 记录并返回模型的详细信息
        _load: 加载 NAS 模型权重

    示例:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")  # 小型模型
        >>> model = NAS("yolo_nas_m")  # 中型模型
        >>> model = NAS("yolo_nas_l")  # 大型模型
        >>> results = model.predict("bus.jpg")

    注意:
        YOLO-NAS 模型仅支持预训练模型，不支持 YAML 配置文件。
    """

    def __init__(self, model: str = "yolo_nas_s.pt") -> None:
        """使用提供的模型或默认模型初始化 NAS 模型。

        参数:
            model (str): 模型名称或权重文件路径，默认 "yolo_nas_s.pt"
                支持的模型: yolo_nas_s, yolo_nas_m, yolo_nas_l

        异常:
            AssertionError: 如果提供了 YAML 配置文件
        """
        # NAS 模型仅支持预训练权重，不支持从配置文件创建
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS 模型仅支持预训练模型。"
        # 调用父类初始化
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """加载现有的 NAS 模型权重或使用预训练权重创建新的 NAS 模型。

        该方法支持两种加载方式：
        1. 从 .pt 文件加载已保存的模型
        2. 从 Super-Gradients 库加载预训练模型

        参数:
            weights (str): 模型权重文件路径或模型名称
                .pt 文件: 加载本地权重文件
                模型名称: 从 Super-Gradients 加载（如 yolo_nas_s）
            task (str, optional): 模型任务类型
        """
        # 导入 Super-Gradients 库（NAS 模型基于此框架）
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            # 从 .pt 文件加载模型
            self.model = torch_load(attempt_download_asset(weights))
        elif suffix == "":
            # 从 Super-Gradients 库加载预训练模型
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # 重写 forward 方法以忽略额外的参数（兼容性处理）
        def new_forward(x, *args, **kwargs):
            """忽略额外的 __call__ 参数，仅使用输入张量。"""
            return self.model._original_forward(x)

        # 保存原始 forward 方法并替换为新方法
        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        # 标准化模型属性以保证兼容性
        self.model.fuse = lambda verbose=True: self.model  # 融合层（空操作）
        self.model.stride = torch.tensor([32])  # 模型步长
        self.model.names = dict(enumerate(self.model._class_names))  # 类别名称字典
        self.model.is_fused = lambda: False  # 用于 info() 方法
        self.model.yaml = {}  # 用于 info() 方法
        self.model.pt_path = weights  # 用于 export() - 权重文件路径
        self.model.task = "detect"  # 用于 export() - 任务类型
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 用于 export() - 配置参数
        self.model.eval()  # 设置为评估模式

    def info(self, detailed: bool = False, verbose: bool = True) -> dict[str, Any]:
        """记录并返回模型信息。

        参数:
            detailed (bool): 是否显示模型的详细信息（层级结构等），默认 False
            verbose (bool): 是否将信息打印到控制台，默认 True

        返回:
            (dict[str, Any]): 包含模型信息的字典

        示例:
            >>> model = NAS("yolo_nas_s")
            >>> info = model.info()  # 获取模型摘要信息
            >>> info = model.info(detailed=True)  # 获取详细信息
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """返回任务到对应预测器和验证器类的映射字典。

        返回:
            (dict[str, dict[str, Any]]): 任务映射字典
                - predictor: NASPredictor - 预测器类
                - validator: NASValidator - 验证器类
        """
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
