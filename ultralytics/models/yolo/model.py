"""
YOLO 模型基础类

该模块提供了 YOLO 系列模型的统一接口,包括:
    - YOLO: 标准 YOLO 模型,支持检测、分割、分类、姿态估计、有向边界框等多种任务
    - YOLOWorld: 开放词汇目标检测模型,支持基于文本描述检测任意类别
    - YOLOE: 增强型 YOLO 模型,支持视觉和文本提示的检测与分割

主要功能:
    - 自动根据模型文件名选择对应的模型类型
    - 统一的训练、验证、预测接口
    - 支持动态类别设置和自定义词汇表
    - 支持视觉提示和文本提示的多模态推理

典型应用场景:
    - 目标检测: 检测图像中的目标并返回边界框
    - 实例分割: 检测目标并生成像素级分割掩码
    - 图像分类: 对整张图像进行分类
    - 姿态估计: 检测人体关键点
    - 有向边界框检测: 检测带旋转角度的目标
    - 开放词汇检测: 基于文本描述检测任意类别
"""

from __future__ import annotations  # 启用延迟类型注解评估,支持 Python 3.9+ 的新式类型提示

from pathlib import Path  # 用于跨平台的路径操作
from typing import Any  # 用于类型提示

import torch  # PyTorch 深度学习框架

# 导入数据加载和推理相关模块
from ultralytics.data.build import load_inference_source  # 加载推理数据源
from ultralytics.engine.model import Model  # YOLO 模型基类
from ultralytics.models import yolo  # YOLO 各任务模块
from ultralytics.nn.tasks import (  # 导入各任务的模型架构
    ClassificationModel,  # 图像分类模型
    DetectionModel,  # 目标检测模型
    OBBModel,  # 有向边界框检测模型
    PoseModel,  # 姿态估计模型
    SegmentationModel,  # 实例分割模型
    WorldModel,  # 开放词汇检测模型
    YOLOEModel,  # YOLOE 检测模型
    YOLOESegModel,  # YOLOE 分割模型
)
from ultralytics.utils import ROOT, YAML  # 工具函数和常量


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, and oriented bounding box detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLO11n detection model
        >>> model = YOLO("yolo11n.pt")

        Load a pretrained YOLO11n segmentation model
        >>> model = YOLO("yolo11n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo11n.yaml")
    """

    def __init__(self, model: str | Path = "yolo11n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types (YOLOWorld or
        YOLOE) based on the model filename.

        初始化 YOLO 模型

        该构造函数会根据模型文件名自动选择合适的模型类型:
            - 文件名包含 "-world": 切换为 YOLOWorld 开放词汇检测模型
            - 文件名包含 "yoloe": 切换为 YOLOE 增强型模型
            - 其他: 使用标准 YOLO 模型

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolo11n.yaml'.
                模型名称或模型文件路径,例如 'yolo11n.pt' (预训练权重) 或 'yolo11n.yaml' (模型配置)
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults
                to auto-detection based on model.
                任务类型,'detect'(检测)/'segment'(分割)/'classify'(分类)/'pose'(姿态)/'obb'(有向框)
                默认根据模型自动检测
            verbose (bool): Display model info on load.
                是否在加载时显示模型信息
        """
        # 将模型路径转换为 Path 对象
        path = Path(model if isinstance(model, (str, Path)) else "")
        # 检查是否为 YOLOWorld 模型 (文件名包含 "-world")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            # 创建 YOLOWorld 实例并替换当前对象
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        # 检查是否为 YOLOE 模型 (文件名包含 "yoloe")
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            # 创建 YOLOE 实例并替换当前对象
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # 使用标准 YOLO 初始化流程
            super().__init__(model=model, task=task, verbose=verbose)
            # 检查是否为 RTDETR 模型 (检测头为 RTDETR)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                # 创建 RTDETR 实例并替换当前对象
                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes.

        任务映射字典

        将任务类型映射到对应的模型、训练器、验证器和预测器类。
        该映射表用于根据任务类型自动选择合适的组件。

        Returns:
            dict: 任务映射字典,包含以下任务:
                - classify: 图像分类任务
                - detect: 目标检测任务
                - segment: 实例分割任务
                - pose: 姿态估计任务
                - obb: 有向边界框检测任务
        """
        return {
            "classify": {  # 图像分类任务
                "model": ClassificationModel,  # 分类模型架构
                "trainer": yolo.classify.ClassificationTrainer,  # 分类训练器
                "validator": yolo.classify.ClassificationValidator,  # 分类验证器
                "predictor": yolo.classify.ClassificationPredictor,  # 分类预测器
            },
            "detect": {  # 目标检测任务
                "model": DetectionModel,  # 检测模型架构
                "trainer": yolo.detect.DetectionTrainer,  # 检测训练器
                "validator": yolo.detect.DetectionValidator,  # 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器
            },
            "segment": {  # 实例分割任务
                "model": SegmentationModel,  # 分割模型架构
                "trainer": yolo.segment.SegmentationTrainer,  # 分割训练器
                "validator": yolo.segment.SegmentationValidator,  # 分割验证器
                "predictor": yolo.segment.SegmentationPredictor,  # 分割预测器
            },
            "pose": {  # 姿态估计任务
                "model": PoseModel,  # 姿态估计模型架构
                "trainer": yolo.pose.PoseTrainer,  # 姿态估计训练器
                "validator": yolo.pose.PoseValidator,  # 姿态估计验证器
                "predictor": yolo.pose.PosePredictor,  # 姿态估计预测器
            },
            "obb": {  # 有向边界框检测任务
                "model": OBBModel,  # OBB 模型架构
                "trainer": yolo.obb.OBBTrainer,  # OBB 训练器
                "validator": yolo.obb.OBBValidator,  # OBB 验证器
                "predictor": yolo.obb.OBBPredictor,  # OBB 预测器
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions without
    requiring training on specific classes. It extends the YOLO architecture to support real-time open-vocabulary
    detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default COCO
        class names.

        初始化 YOLOv8-World 模型

        加载一个开放词汇目标检测模型。如果没有提供自定义类别名称,则使用默认的 COCO 类别名称。

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
                预训练模型文件路径,支持 *.pt (权重) 和 *.yaml (配置) 格式
            verbose (bool): If True, prints additional information during initialization.
                是否在初始化时打印详细信息
        """
        # 调用父类初始化,固定任务类型为 "detect"
        super().__init__(model=model, task="detect", verbose=verbose)

        # 如果模型没有类别名称属性,则分配默认的 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, validator, and predictor classes.

        任务映射字典

        YOLOWorld 仅支持目标检测任务,使用专门的 WorldModel 和 WorldTrainer。

        Returns:
            dict: 任务映射字典,仅包含 detect 任务
        """
        return {
            "detect": {  # 开放词汇目标检测任务
                "model": WorldModel,  # World 模型架构
                "validator": yolo.detect.DetectionValidator,  # 使用标准检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 使用标准检测预测器
                "trainer": yolo.world.WorldTrainer,  # World 专用训练器
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """Set the model's class names for detection.

        设置模型的检测类别名称

        为 YOLOWorld 模型动态设置检测类别,支持开放词汇检测。
        模型会根据提供的类别名称生成对应的文本嵌入。

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
                类别名称列表,例如 ["person", "car", "dog"]
        """
        # 调用模型的 set_classes 方法设置类别 (会生成文本嵌入)
        self.model.set_classes(classes)
        # 移除背景类别 (如果存在)
        background = " "
        if background in classes:
            classes.remove(background)
        # 更新模型的类别名称属性
        self.model.names = classes

        # 如果预测器已初始化,同步更新预测器的类别名称
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """Initialize YOLOE model with a pre-trained model file.

        初始化 YOLOE 模型

        加载一个增强型 YOLO 模型,支持视觉提示和文本提示的检测与分割。

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
                预训练模型文件路径,支持 *.pt (权重) 和 *.yaml (配置) 格式
            task (str, optional): Task type for the model. Auto-detected if None.
                任务类型,'detect' 或 'segment',默认自动检测
            verbose (bool): If True, prints additional information during initialization.
                是否在初始化时打印详细信息
        """
        # 调用父类初始化
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, validator, and predictor classes.

        任务映射字典

        YOLOE 支持检测和分割两种任务,每种任务使用专门的模型架构和组件。

        Returns:
            dict: 任务映射字典,包含 detect 和 segment 任务
        """
        return {
            "detect": {  # YOLOE 检测任务
                "model": YOLOEModel,  # YOLOE 检测模型架构
                "validator": yolo.yoloe.YOLOEDetectValidator,  # YOLOE 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 标准检测预测器
                "trainer": yolo.yoloe.YOLOETrainer,  # YOLOE 检测训练器
            },
            "segment": {  # YOLOE 分割任务
                "model": YOLOESegModel,  # YOLOE 分割模型架构
                "validator": yolo.yoloe.YOLOESegValidator,  # YOLOE 分割验证器
                "predictor": yolo.segment.SegmentationPredictor,  # 标准分割预测器
                "trainer": yolo.yoloe.YOLOESegTrainer,  # YOLOE 分割训练器
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts.

        获取文本位置编码

        根据输入的文本列表生成对应的文本嵌入向量,用于文本提示检测。

        Args:
            texts (list[str]): 文本列表,例如类别名称

        Returns:
            torch.Tensor: 文本位置编码张量
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """获取图像和视觉特征的视觉位置编码

        该方法根据输入图像从视觉特征中提取位置编码。模型必须是 YOLOEModel 的实例。

        参数:
            img (torch.Tensor): 输入图像张量
            visual (torch.Tensor): 从图像中提取的视觉特征

        返回:
            (torch.Tensor): 视觉位置编码

        示例:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """为 YOLOE 模型设置词汇表和类别名称

        该方法配置模型用于文本处理和分类任务的词汇表和类别名称。模型必须是 YOLOEModel 的实例。

        参数:
            vocab (list[str]): 词汇表列表,包含模型用于文本处理的标记或单词
            names (list[str]): 模型可以检测或分类的类别名称列表

        异常:
            AssertionError: 如果模型不是 YOLOEModel 的实例

        示例:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names.

        获取类别名称的词汇表

        根据提供的类别名称获取模型的词汇表映射。

        Args:
            names (list[str]): 类别名称列表

        Returns:
            list[str]: 词汇表列表
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """Set the model's class names and embeddings for detection.

        设置模型的类别名称和嵌入向量

        为 YOLOE 模型设置检测类别及其对应的嵌入向量。如果未提供嵌入向量,
        会自动从类别名称生成文本嵌入。

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
                类别名称列表,例如 ["person", "car", "dog"]
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
                与类别对应的嵌入向量,如果为 None 则自动生成
        """
        assert isinstance(self.model, YOLOEModel)
        # 如果未提供嵌入向量,则从类别名称生成文本嵌入
        if embeddings is None:
            embeddings = self.get_text_pe(classes)
        # 调用模型的 set_classes 方法设置类别和嵌入
        self.model.set_classes(classes, embeddings)
        # 验证不存在背景类别
        assert " " not in classes
        # 更新模型的类别名称属性
        self.model.names = classes

        # 如果预测器已初始化,同步更新预测器的类别名称
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """Validate the model using text or visual prompts.

        使用文本或视觉提示验证模型

        该方法支持两种验证模式:
            - 文本提示模式 (load_vp=False): 使用文本嵌入作为类别提示
            - 视觉提示模式 (load_vp=True): 使用参考图像中的视觉特征作为提示

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
                验证器函数,如果为 None 则加载默认验证器
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
                是否加载视觉提示,False 表示使用文本提示
            refer_data (str, optional): Path to the reference data for visual prompts.
                视觉提示的参考数据路径
            **kwargs (Any): Additional keyword arguments to override default settings.
                额外的关键字参数,用于覆盖默认设置

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
                验证统计信息,包含验证过程中计算的各项指标
        """
        # 设置方法默认参数 (视觉提示模式不使用矩形推理)
        custom = {"rect": not load_vp}  # method defaults
        # 合并参数,优先级从低到高: overrides < custom < kwargs
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        # 加载验证器并执行验证
        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        # 保存验证指标
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, list] = {},
        refer_image=None,
        predictor=yolo.yoloe.YOLOEVPDetectPredictor,
        **kwargs,
    ):
        """Run prediction on images, videos, directories, streams, etc.

        使用视觉提示或文本提示进行预测

        该方法支持两种预测模式:
            1. 标准模式: 直接使用模型的类别进行检测
            2. 视觉提示模式: 使用参考图像中标注的目标作为检测模板

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths, directory
                paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
                预测数据源,支持图像路径、目录、URL、PIL图像、numpy数组或摄像头索引
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a generator as they
                are computed.
                是否流式返回预测结果,True 表示返回生成器
            visual_prompts (dict[str, list]): Dictionary containing visual prompts for the model. Must include 'bboxes'
                and 'cls' keys when non-empty.
                视觉提示字典,包含 'bboxes' (边界框列表) 和 'cls' (类别列表) 键
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
                视觉提示的参考图像
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically loaded
                based on the task.
                自定义预测器,默认使用 YOLOEVPDetectPredictor
            **kwargs (Any): Additional keyword arguments passed to the predictor.
                传递给预测器的额外参数

        Returns:
            (list | generator): List of Results objects or generator of Results objects if stream=True.
                预测结果列表或生成器 (stream=True 时)

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        # 如果提供了视觉提示,则进入视觉提示预测模式
        if len(visual_prompts):
            # 验证视觉提示格式:必须包含 'bboxes' 和 'cls' 键
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            # 验证边界框和类别数量匹配
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            # 如果预测器类型不匹配,则创建新的视觉提示预测器
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides["imgsz"]),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # 表示多张图像
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # 注意: 将第一帧设置为视频/流推理的参考图像
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # 重置预测器
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # 如果没有视觉提示则重置预测器

        return super().predict(source, stream, **kwargs)
