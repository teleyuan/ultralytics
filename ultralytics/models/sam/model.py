# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset

---

SAM 模型接口模块

该模块提供了 Segment Anything Model (SAM) 的 Ultralytics 实现接口。
SAM 是 Meta AI 开发的通用图像分割模型，能够根据各种提示（点、框、掩码）进行实时分割。

核心特性:
    - 提示式分割: 支持点、边界框、掩码等多种提示方式
    - 实时性能: 优化的推理速度，适合实时应用
    - 零样本迁移: 无需额外训练即可适应新的图像分布和任务
    - SA-1B 数据集: 在包含 1100 万张图像的 SA-1B 数据集上训练

技术特点:
    - 基于 Vision Transformer (ViT) 的图像编码器
    - 轻量级的掩码解码器
    - 灵活的提示编码器，支持多种提示类型
    - 支持 SAM、SAM2 和 SAM3 多个版本
"""

from __future__ import annotations  # 启用延迟类型注解评估

from pathlib import Path  # 用于路径操作

# 导入基础模型类和工具函数
from ultralytics.engine.model import Model  # Ultralytics 模型基类
from ultralytics.utils.torch_utils import model_info  # 模型信息打印工具

# 导入 SAM 各版本的预测器
from .predict import Predictor, SAM2Predictor, SAM3Predictor


class SAM(Model):
    """SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

    This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for promptable
    segmentation with versatility in image analysis. It supports various prompts such as bounding boxes, points, or
    labels, and features zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM model.
        is_sam2 (bool): Indicates whether the model is SAM2 variant.
        task (str): The task type, set to "segment" for SAM models.

    Methods:
        predict: Perform segmentation prediction on the given image or video source.
        info: Log information about the SAM model.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"Detected {len(r.masks)} masks")

    ---

    SAM (Segment Anything Model) 接口类，用于实时图像分割任务

    该类提供了 SAM 模型的统一接口，支持提示式分割。可以使用点、边界框或标签作为提示，
    模型会生成对应的分割掩码。具有零样本学习能力，无需微调即可应用于新场景。

    属性:
        model (torch.nn.Module): 加载的 SAM 模型实例
        is_sam2 (bool): 标识是否为 SAM2 版本
        is_sam3 (bool): 标识是否为 SAM3 版本
        task (str): 任务类型，对于 SAM 模型固定为 "segment"

    方法:
        predict: 对给定的图像或视频源执行分割预测
        info: 输出 SAM 模型的详细信息
        __call__: predict 方法的别名，提供更便捷的调用方式

    示例:
        >>> sam = SAM("sam_b.pt")  # 加载 SAM Base 模型
        >>> results = sam.predict("image.jpg", points=[[500, 375]])  # 使用点提示进行分割
        >>> for r in results:
        ...     print(f"检测到 {len(r.masks)} 个掩码")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """Initialize the SAM (Segment Anything Model) instance.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.

        ---

        初始化 SAM (Segment Anything Model) 实例

        Args:
            model (str): 预训练 SAM 模型文件的路径，文件扩展名应为 .pt 或 .pth
                默认: "sam_b.pt" (SAM Base 模型)
                可选: "sam_l.pt" (Large), "sam_h.pt" (Huge), "mobile_sam.pt" (移动端)

        Raises:
            NotImplementedError: 如果模型文件扩展名不是 .pt 或 .pth
        """
        # 验证模型文件格式
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        # 检测模型版本（通过文件名判断）
        self.is_sam2 = "sam2" in Path(model).stem  # 是否为 SAM2 版本
        self.is_sam3 = "sam3" in Path(model).stem  # 是否为 SAM3 版本
        # 调用父类初始化，设置任务类型为分割
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """Load the specified weights into the SAM model.

        Args:
            weights (str): Path to the weights file. Should be a .pt or .pth file containing the model parameters.
            task (str | None): Task name. If provided, it specifies the particular task the model is being loaded for.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")

        ---

        加载指定的权重文件到 SAM 模型

        根据模型版本（SAM3 或 SAM/SAM2）选择相应的构建函数加载模型权重。
        SAM3 使用 build_interactive_sam3，其他版本使用 build_sam。

        Args:
            weights (str): 权重文件路径，应为包含模型参数的 .pt 或 .pth 文件
            task (str | None): 任务名称（可选），指定模型加载的特定任务

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")  # 加载自定义权重
        """
        if self.is_sam3:
            # SAM3 版本：使用专用的交互式构建函数
            from .build_sam3 import build_interactive_sam3

            self.model = build_interactive_sam3(weights)
        else:
            # SAM/SAM2 版本：使用标准构建函数（首次导入较慢）
            from .build import build_sam  # slow import

            self.model = build_sam(weights)

    def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """Perform segmentation prediction on the given image or video source.

        Args:
            source (str | PIL.Image | np.ndarray): Path to the image or video file, or a PIL.Image object, or a
                np.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (list[list[float]] | None): List of points for prompted segmentation.
            labels (list[int] | None): List of labels for prompted segmentation.
            **kwargs (Any): Additional keyword arguments for prediction.

        Returns:
            (list): The model predictions.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"Detected {len(r.masks)} masks")

        ---

        对给定的图像或视频源执行分割预测

        该方法支持多种提示方式进行分割：点提示、边界框提示或标签提示。
        模型会根据提示生成对应的分割掩码。

        Args:
            source (str | PIL.Image | np.ndarray): 图像或视频文件路径，或 PIL.Image 对象，或 numpy 数组
            stream (bool): 是否启用实时流式处理，默认 False
            bboxes (list[list[float]] | None): 边界框坐标列表，格式为 [[x1,y1,x2,y2], ...]
            points (list[list[float]] | None): 点坐标列表，格式为 [[x,y], ...]
            labels (list[int] | None): 标签列表，用于指定点的类型（前景/背景）
            **kwargs (Any): 其他预测参数（如 device, save, show 等）

        Returns:
            (list): 模型预测结果列表，每个结果包含分割掩码、置信度等信息

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> # 使用点提示
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> # 使用边界框提示
            >>> results = sam.predict("image.jpg", bboxes=[[100, 100, 500, 500]])
            >>> for r in results:
            ...     print(f"检测到 {len(r.masks)} 个掩码")
        """
        # 设置默认参数
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        # 合并用户参数（用户参数优先）
        kwargs = {**overrides, **kwargs}
        # 组织提示信息
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        # 调用父类的 predict 方法
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """Perform segmentation prediction on the given image or video source.

        This method is an alias for the 'predict' method, providing a convenient way to call the SAM model for
        segmentation tasks.

        Args:
            source (str | PIL.Image | np.ndarray | None): Path to the image or video file, or a PIL.Image object, or a
                np.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (list[list[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (list[list[float]] | None): List of points for prompted segmentation.
            labels (list[int] | None): List of labels for prompted segmentation.
            **kwargs (Any): Additional keyword arguments to be passed to the predict method.

        Returns:
            (list): The model predictions, typically containing segmentation masks and other relevant information.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"Detected {len(results[0].masks)} masks")

        ---

        执行图像或视频源的分割预测（predict 方法的便捷别名）

        该方法是 predict 方法的别名，允许直接调用模型实例来执行分割。
        这种调用方式更加简洁直观。

        Args:
            source (str | PIL.Image | np.ndarray | None): 图像或视频文件路径，或图像对象
            stream (bool): 是否启用实时流式处理
            bboxes (list[list[float]] | None): 边界框坐标列表
            points (list[list[float]] | None): 点坐标列表
            labels (list[int] | None): 标签列表
            **kwargs (Any): 传递给 predict 方法的其他关键字参数

        Returns:
            (list): 模型预测结果，包含分割掩码和相关信息

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])  # 直接调用
            >>> print(f"检测到 {len(results[0].masks)} 个掩码")
        """
        # 委托给 predict 方法
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed: bool = False, verbose: bool = True):
        """Log information about the SAM model.

        Args:
            detailed (bool): If True, displays detailed information about the model layers and operations.
            verbose (bool): If True, prints the information to the console.

        Returns:
            (tuple): A tuple containing the model's information (string representations of the model).

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # Print summary information

        ---

        输出 SAM 模型的详细信息

        Args:
            detailed (bool): 是否显示详细的层级和操作信息，默认 False
            verbose (bool): 是否将信息打印到控制台，默认 True

        Returns:
            (tuple): 包含模型信息的元组（模型的字符串表示）

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()  # 输出模型摘要信息
            >>> info = sam.info(detailed=True)  # 输出详细信息
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
        """Provide a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (dict[str, dict[str, Type[Predictor]]]): A dictionary mapping the 'segment' task to its corresponding
                Predictor class. For SAM2 models, it maps to SAM2Predictor, otherwise to the standard Predictor.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}

        ---

        提供从 'segment' 任务到对应预测器的映射关系

        根据 SAM 模型的版本（SAM、SAM2 或 SAM3）返回相应的预测器类。
        这个映射用于自动选择正确的预测器来处理分割任务。

        Returns:
            (dict[str, dict[str, Type[Predictor]]]): 任务到预测器的映射字典
                - SAM3: 使用 SAM3Predictor
                - SAM2: 使用 SAM2Predictor
                - SAM: 使用基础 Predictor

        示例:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)  # {'segment': {'predictor': <class 'Predictor'>}}
        """
        return {
            "segment": {"predictor": SAM2Predictor if self.is_sam2 else SAM3Predictor if self.is_sam3 else Predictor}
        }
