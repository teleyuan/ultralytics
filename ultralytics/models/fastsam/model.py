"""
FastSAM 模型接口模块

FastSAM (Fast Segment Anything Model) 是基于 YOLOv8 架构的快速分割模型，
提供了与 SAM 相似的分割能力，但速度快得多，更适合实时应用。

主要特点:
    - 基于 YOLO 的实时分割架构
    - 支持提示式分割（点、框、文本）
    - 比原始 SAM 快 50 倍以上
    - 适合边缘设备和移动端部署
    - 端到端训练，无需复杂的后处理
"""

from __future__ import annotations  # 启用延迟类型注解评估

from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

from ultralytics.engine.model import Model  # 基础模型类

# 导入 FastSAM 组件
from .predict import FastSAMPredictor  # FastSAM 预测器
from .val import FastSAMValidator  # FastSAM 验证器


class FastSAM(Model):
    """FastSAM 模型接口类，用于 Segment Anything 任务

    该类扩展了基础 Model 类，为 FastSAM（快速 Segment Anything 模型）提供特定功能。
    FastSAM 支持可选的提示输入，可以高效准确地进行图像分割。

    属性:
        model (str): 预训练 FastSAM 模型文件的路径
        task (str): 任务类型，对于 FastSAM 模型固定为 "segment"

    方法:
        predict: 对图像或视频源执行分割预测，支持可选提示
        task_map: 返回分割任务到预测器和验证器类的映射

    示例:
        初始化 FastSAM 模型并运行预测
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("FastSAM-x.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

        使用边界框提示运行预测
        >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
    """

    def __init__(self, model: str | Path = "FastSAM-x.pt"):
        """使用指定的预训练权重初始化 FastSAM 模型。

        参数:
            model (str | Path): 预训练模型文件路径，默认 "FastSAM-x.pt"
                支持的模型: FastSAM-s.pt (小型), FastSAM-x.pt (大型)

        异常:
            AssertionError: 如果提供了 YAML 配置文件（FastSAM 仅支持预训练权重）
        """
        # 兼容性处理：将旧的模型名称映射到新名称
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        # FastSAM 仅支持预训练权重，不支持从 YAML 配置文件创建
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM 仅支持预训练权重。"
        # 调用父类初始化，设置任务类型为分割
        super().__init__(model=model, task="segment")

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes: list | None = None,
        points: list | None = None,
        labels: list | None = None,
        texts: list | None = None,
        **kwargs: Any,
    ):
        """对图像或视频源执行分割预测。

        支持使用边界框、点、标签和文本进行提示式分割。该方法将这些提示打包后
        传递给父类的 predict 方法进行处理。

        参数:
            source (str | PIL.Image | np.ndarray): 预测输入源，可以是文件路径、URL、PIL 图像或 numpy 数组
            stream (bool): 是否启用视频输入的实时流式模式，默认 False
            bboxes (list, optional): 边界框坐标列表，格式为 [[x1, y1, x2, y2]]
            points (list, optional): 点坐标列表，格式为 [[x, y]]
            labels (list, optional): 分割的类别标签
            texts (list, optional): 文本提示，用于指导分割
            **kwargs (Any): 传递给预测器的其他关键字参数

        返回:
            (list): 包含预测结果的 Results 对象列表

        示例:
            >>> model = FastSAM("FastSAM-x.pt")
            >>> # 无提示分割
            >>> results = model.predict("image.jpg")
            >>> # 使用边界框提示
            >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
            >>> # 使用点和文本提示
            >>> results = model.predict("image.jpg", points=[[150, 150]], texts=["dog"])
        """
        # 将所有提示打包为字典
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        # 调用父类的 predict 方法，传递提示
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """返回分割任务到对应预测器和验证器类的映射字典。

        返回:
            (dict[str, dict[str, Any]]): 任务映射字典，包含预测器和验证器
        """
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
