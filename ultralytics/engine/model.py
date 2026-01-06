"""
YOLO 模型基类模块

该模块提供了 YOLO 模型的统一接口,支持训练、验证、预测、导出等核心功能。
它是所有 YOLO 变体(检测、分割、分类、姿态估计等)的基础类。
"""

from __future__ import annotations  # 启用延迟类型注解评估

import inspect  # 用于检查调用栈和函数签名
from pathlib import Path  # 跨平台路径操作
from typing import Any  # 类型提示

import numpy as np  # 数组操作
import torch  # PyTorch 深度学习框架
from PIL import Image  # 图像处理

# 配置相关导入
from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
# 结果处理
from ultralytics.engine.results import Results
# 神经网络任务相关
from ultralytics.nn.tasks import guess_model_task, load_checkpoint, yaml_model_load
# 工具函数
from ultralytics.utils import (
    ARGV,  # 命令行参数
    ASSETS,  # 资源文件路径
    DEFAULT_CFG_DICT,  # 默认配置字典
    LOGGER,  # 日志记录器
    RANK,  # 分布式训练进程编号
    SETTINGS,  # 全局设置
    YAML,  # YAML 配置处理
    callbacks,  # 回调函数管理
    checks,  # 检查和验证函数
)


class Model(torch.nn.Module):
    """实现 YOLO 模型的基类，统一不同模型类型的 API。

    该类为 YOLO 模型的各种操作提供统一接口，如训练、验证、预测、导出和基准测试。
    它处理不同类型的模型，包括从本地文件、Ultralytics HUB 或 Triton Server 加载的模型。

    属性:
        callbacks (dict): 模型操作期间各种事件的回调函数字典。
        predictor (BasePredictor): 用于进行预测的预测器对象。
        model (torch.nn.Module): 底层的 PyTorch 模型。
        trainer (BaseTrainer): 用于训练模型的训练器对象。
        ckpt (dict): 如果模型从 *.pt 文件加载，则为检查点数据。
        cfg (str): 如果从 *.yaml 文件加载，则为模型的配置。
        ckpt_path (str): 检查点文件的路径。
        overrides (dict): 模型配置覆盖参数的字典。
        metrics (dict): 最新的训练/验证指标。
        session (HUBTrainingSession): Ultralytics HUB 会话（如果适用）。
        task (str): 模型预期执行的任务类型。
        model_name (str): 模型的名称。

    方法:
        __call__: predict 方法的别名，使模型实例可调用。
        _new: 基于配置文件初始化新模型。
        _load: 从检查点文件加载模型。
        _check_is_pytorch_model: 确保模型是 PyTorch 模型。
        reset_weights: 将模型权重重置为初始状态。
        load: 从指定文件加载模型权重。
        save: 将模型的当前状态保存到文件。
        info: 记录或返回模型信息。
        fuse: 融合 Conv2d 和 BatchNorm2d 层以优化推理。
        predict: 执行目标检测预测。
        track: 执行目标跟踪。
        val: 在数据集上验证模型。
        benchmark: 在各种导出格式上对模型进行基准测试。
        export: 将模型导出为不同格式。
        train: 在数据集上训练模型。
        tune: 执行超参数调优。
        _apply: 将函数应用于模型的张量。
        add_callback: 为事件添加回调函数。
        clear_callback: 清除事件的所有回调。
        reset_callbacks: 将所有回调重置为默认函数。

    示例:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: str | Path | Model = "yolo11n.pt",
        task: str | None = None,
        verbose: bool = False,
    ) -> None:
        """初始化 YOLO 模型类的新实例。

        此构造函数基于提供的模型路径或名称设置模型。它处理各种类型的模型来源，
        包括本地文件、Ultralytics HUB 模型和 Triton Server 模型。该方法初始化
        模型的几个重要属性，并为训练、预测或导出等操作做准备。

        参数:
            model (str | Path | Model): 要加载或创建的模型的路径或名称。可以是本地文件路径、
                来自 Ultralytics HUB 的模型名称、Triton Server 模型或已初始化的 Model 实例。
            task (str, optional): 模型的特定任务。如果为 None，将从配置中推断。
            verbose (bool): 如果为 True，在模型初始化和后续操作期间启用详细输出。

        异常:
            FileNotFoundError: 如果指定的模型文件不存在或无法访问。
            ValueError: 如果模型文件或配置无效或不受支持。
            ImportError: 如果未安装特定模型类型（如 HUB SDK）所需的依赖项。
        """
        # 如果传入的已经是一个 Model 实例,直接复制其属性
        if isinstance(model, Model):
            self.__dict__ = model.__dict__  # 接受已初始化的 Model 对象
            return

        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 初始化回调函数字典,用于在训练/推理的各个阶段执行自定义操作
        self.callbacks = callbacks.get_default_callbacks()

        # 初始化各个组件为 None,按需懒加载
        self.predictor = None  # 预测器对象(可复用,避免重复初始化)
        self.model = None  # 模型结构和权重
        self.trainer = None  # 训练器对象

        # 模型加载相关属性
        self.ckpt = {}  # 如果从 *.pt 文件加载,存储检查点数据，包含模型结构和权重，优化器状态等
        self.cfg = None  # 如果从 *.yaml 文件加载,存储配置
        self.ckpt_path = None  # 检查点文件路径

        # 配置和元数据
        self.overrides = {}  # 覆盖默认配置的参数字典
        self.metrics = None  # 验证/训练指标
        self.session = None  # Ultralytics HUB 会话对象
        self.task = task  # 任务类型: detect/segment/classify/pose 等
        self.model_name = None  # 模型名称或路径

        # 标准化模型路径字符串(去除首尾空格)
        model = str(model).strip()

        # 检查是否是 Ultralytics HUB 模型 (来自 https://hub.ultralytics.com)
        if self.is_hub_model(model):
            from ultralytics.hub import HUBTrainingSession

            # 从 HUB 获取模型
            checks.check_requirements("hub-sdk>=0.0.12")  # 确保安装了 HUB SDK
            session = HUBTrainingSession.create_session(model)  # 创建 HUB 会话
            model = session.model_file  # 获取下载的模型文件路径
            if session.train_args:  # 如果 HUB 发送了训练参数
                self.session = session  # 保存会话对象用于后续训练

        # 检查是否是 Triton Server 模型 (用于高性能推理服务)
        elif self.is_triton_model(model):
            self.model_name = self.model = model  # 保存 Triton Server URL
            self.overrides["task"] = task or "detect"  # 如果未指定任务,默认设为检测
            return  # Triton 模型不需要进一步初始化

        # 加载或创建新的 YOLO 模型
        # 设置 CUBLAS 工作区配置,避免确定性计算警告(适用于可复现实验)
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # 根据文件扩展名选择加载方式
        if str(model).endswith((".yaml", ".yml")):
            # 从配置文件创建新模型
            self._new(model, task=task, verbose=verbose)
        else:
            # 从权重文件加载已训练模型
            self._load(model, task=task)

        # 删除继承自 nn.Module 的 training 属性,以便直接访问 self.model.training
        # 这样可以正确控制模型的训练/评估模式
        del self.training

    def __call__(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """predict 方法的别名，使模型实例可调用以进行预测。

        此方法通过允许直接使用所需参数调用模型实例来简化预测过程。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): 要进行预测的图像源。
                可以是文件路径、URL、PIL 图像、numpy 数组、PyTorch 张量或这些类型的列表/元组。
            stream (bool): 如果为 True，将输入源视为连续流进行预测。
            **kwargs (Any): 用于配置预测过程的其他关键字参数。

        返回:
            (list[ultralytics.engine.results.Results]): 预测结果列表，每个结果封装在 Results 对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """检查给定的模型字符串是否为 Triton Server URL。

        此静态方法通过使用 urllib.parse.urlsplit() 解析其组件来确定提供的模型字符串
        是否表示有效的 Triton Server URL。

        参数:
            model (str): 要检查的模型字符串。

        返回:
            (bool): 如果模型字符串是有效的 Triton Server URL，则为 True，否则为 False。

        示例:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            True
            >>> Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """检查提供的模型是否为 Ultralytics HUB 模型。

        此静态方法确定给定的模型字符串是否表示有效的 Ultralytics HUB 模型标识符。

        参数:
            model (str): 要检查的模型字符串。

        返回:
            (bool): 如果模型是有效的 Ultralytics HUB 模型，则为 True，否则为 False。

        示例:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/MODEL")
            True
            >>> Model.is_hub_model("yolo11n.pt")
            False
        """
        from ultralytics.hub import HUB_WEB_ROOT

        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """初始化新模型并从模型定义中推断任务类型。

        基于提供的配置文件创建新的模型实例。加载模型配置，如果未指定则推断任务类型，
        并使用任务映射中的适当类初始化模型。

        参数:
            cfg (str): YAML 格式的模型配置文件路径。
            task (str, optional): 模型的特定任务。如果为 None，将从配置中推断。
            model (torch.nn.Module, optional): 自定义模型实例。如果提供，将使用它而不是创建新的。
            verbose (bool): 如果为 True，在加载期间显示模型信息。

        异常:
            ValueError: 如果配置文件无效或无法推断任务。
            ImportError: 如果未安装指定任务所需的依赖项。

        示例:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # 以下添加以允许从 YAML 导出
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 组合默认参数和模型参数（优先使用模型参数）
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """从检查点文件加载模型或从权重文件初始化模型。

        此方法处理从 .pt 检查点文件或其他权重文件格式加载模型。它根据加载的权重
        设置模型、任务和相关属性。

        参数:
            weights (str): 要加载的模型权重文件路径。
            task (str, optional): 与模型关联的任务。如果为 None，将从模型中推断。

        异常:
            FileNotFoundError: 如果指定的权重文件不存在或无法访问。
            ValueError: 如果权重文件格式不受支持或无效。

        示例:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # 下载并返回本地文件
        weights = checks.check_model_file_from_stem(weights)  # 添加后缀，即 yolo11n -> yolo11n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # 在所有情况下运行，与上面的调用不重复
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """检查模型是否为 PyTorch 模型，如果不是则引发 TypeError。

        此方法验证模型是否为 PyTorch 模块或 .pt 文件。它用于确保某些需要 PyTorch 模型
        的操作仅在兼容的模型类型上执行。

        异常:
            TypeError: 如果模型不是 PyTorch 模块或 .pt 文件。错误消息提供有关支持的
                模型格式和操作的详细信息。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # 不会引发错误
            >>> model = Model("yolo11n.onnx")
            >>> model._check_is_pytorch_model()  # 引发 TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> Model:
        """将模型的权重重置为初始状态。

        此方法遍历模型中的所有模块，如果它们具有 'reset_parameters' 方法，则重置其参数。
        它还确保所有参数的 'requires_grad' 设置为 True，使它们能够在训练期间更新。

        返回:
            (Model): 权重已重置的类实例。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: str | Path = "yolo11n.pt") -> Model:
        """从指定的权重文件加载参数到模型中。

        此方法支持从文件或直接从权重对象加载权重。它按名称和形状匹配参数并将它们传输到模型。

        参数:
            weights (str | Path): 权重文件的路径或权重对象。

        返回:
            (Model): 已加载权重的类实例。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # 记住权重以用于 DDP 训练
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """将当前模型状态保存到文件。

        此方法将模型的检查点（ckpt）导出到指定的文件名。它包括元数据，例如日期、
        Ultralytics 版本、许可证信息和文档链接。

        参数:
            filename (str | Path): 要保存模型的文件名。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """显示模型信息。

        此方法根据传递的参数提供模型的概述或详细信息。它可以控制输出的详细程度
        并将信息作为列表返回。

        参数:
            detailed (bool): 如果为 True，显示有关模型层和参数的详细信息。
            verbose (bool): 如果为 True，打印信息。如果为 False，将信息作为列表返回。

        返回:
            (list[str]): 包含有关模型的各种信息的字符串列表，包括模型摘要、层详细信息
                和参数计数。如果 verbose 为 True 则为空。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # 打印模型摘要
            >>> info_list = model.info(detailed=True, verbose=False)  # 将详细信息作为列表返回
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        """融合模型中的 Conv2d 和 BatchNorm2d 层以优化推理。

        此方法遍历模型的模块，并将连续的 Conv2d 和 BatchNorm2d 层融合为单个层。
        这种融合可以通过减少前向传递期间所需的操作数和内存访问次数来显著提高推理速度。

        融合过程通常涉及将 BatchNorm2d 参数（均值、方差、权重和偏置）折叠到前面的
        Conv2d 层的权重和偏置中。这将产生一个在一步中执行卷积和归一化的单个 Conv2d 层。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # 模型现已融合并准备好进行优化推理
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """基于提供的源生成图像嵌入。

        此方法是 'predict()' 方法的包装器，专注于从图像源生成嵌入。
        它允许通过各种关键字参数自定义嵌入过程。

        参数:
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor): 用于生成嵌入的图像源。
                可以是文件路径、URL、PIL 图像、numpy 数组等。
            stream (bool): 如果为 True，则流式传输预测。
            **kwargs (Any): 用于配置嵌入过程的其他关键字参数。

        返回:
            (list[torch.Tensor]): 包含图像嵌入的列表。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # 如果未传递索引，则嵌入倒数第二层
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        """使用 YOLO 模型对给定的图像源执行预测。

        此方法通过关键字参数促进预测过程，允许各种配置。它支持使用自定义预测器或
        默认预测器方法进行预测。该方法处理不同类型的图像源，并可以在流模式下运行。

        参数:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): 要进行预测的图像源。
                接受各种类型，包括文件路径、URL、PIL 图像、numpy 数组和 torch 张量。
            stream (bool): 如果为 True，将输入源视为连续流进行预测。
            predictor (BasePredictor, optional): 用于进行预测的自定义预测器类的实例。
                如果为 None，该方法使用默认预测器。
            **kwargs (Any): 用于配置预测过程的其他关键字参数。

        返回:
            (list[ultralytics.engine.results.Results]): 预测结果列表，每个结果封装在 Results 对象中。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # 打印检测边界框

        注意:
            - 如果未提供 'source'，它默认为 ASSETS 常量并发出警告。
            - 该方法在尚未存在时设置新的预测器，并在每次调用时更新其参数。
            - 对于 SAM 类型的模型，可以将 'prompts' 作为关键字参数传递。
        """
        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}  # 方法默认值
        args = {**self.overrides, **custom, **kwargs}  # 右侧的参数优先级最高
        prompts = args.pop("prompts", None)  # 用于 SAM 类型的模型

        if not self.predictor or self.predictor.args.device != args.get("device", self.predictor.args.device):
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # 仅在预测器已设置时更新参数
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # 用于 SAM 类型的模型
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[Results]:
        """使用注册的跟踪器对指定的输入源进行目标跟踪。

        此方法使用模型的预测器和可选的注册跟踪器执行目标跟踪。它处理各种输入源，
        例如文件路径或视频流，并支持通过关键字参数进行自定义。该方法在尚未存在时
        注册跟踪器，并可以在调用之间持久化它们。

        参数:
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor, optional): 目标跟踪的输入源。
                可以是文件路径、URL 或视频流。
            stream (bool): 如果为 True，将输入源视为连续视频流。
            persist (bool): 如果为 True，在对此方法的不同调用之间持久化跟踪器。
            **kwargs (Any): 用于配置跟踪过程的其他关键字参数。

        返回:
            (list[ultralytics.engine.results.Results]): 跟踪结果列表，每个都是 Results 对象。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # 打印跟踪 ID

        注意:
            - 此方法为基于 ByteTrack 的跟踪设置默认置信度阈值 0.1。
            - 跟踪模式在关键字参数中显式设置。
            - 视频跟踪的批次大小设置为 1。
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # 基于 ByteTrack 的方法需要低置信度预测作为输入
        kwargs["batch"] = kwargs.get("batch") or 1  # 视频跟踪的批次大小为 1
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """使用指定的数据集和验证配置验证模型。

        此方法通过各种设置促进模型验证过程，允许自定义。它支持使用自定义验证器或
        默认验证方法进行验证。该方法结合默认配置、方法特定的默认值和用户提供的参数
        来配置验证过程。

        参数:
            validator (ultralytics.engine.validator.BaseValidator, optional): 用于验证模型的自定义验证器类的实例。
            **kwargs (Any): 用于自定义验证过程的任意关键字参数。

        返回:
            (ultralytics.utils.metrics.DetMetrics): 从验证过程获得的验证指标。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # 打印 mAP50-95
        """
        custom = {"rect": True}  # 方法默认值
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 右侧的参数优先级最高

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        """在各种导出格式中对模型进行基准测试以评估性能。

        此方法评估模型在不同导出格式（如 ONNX、TorchScript 等）中的性能。它使用
        ultralytics.utils.benchmarks 模块中的 'benchmark' 函数。基准测试使用默认
        配置值、模型特定参数、方法特定默认值和任何其他用户提供的关键字参数的组合来配置。

        参数:
            data (str): 用于基准测试的数据集路径。
            verbose (bool): 是否打印详细的基准测试信息。
            format (str): 用于特定基准测试的导出格式名称。
            **kwargs (Any): 用于自定义基准测试过程的任意关键字参数。常用选项包括:
                - imgsz (int | list[int]): 用于基准测试的图像大小。
                - half (bool): 是否使用半精度（FP16）模式。
                - int8 (bool): 是否使用 int8 精度模式。
                - device (str): 运行基准测试的设备（例如 'cpu'、'cuda'）。

        返回:
            (dict): 包含基准测试过程结果的字典，包括不同导出格式的指标。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        from .exporter import export_formats

        custom = {"verbose": False}  # 方法默认值
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format, [])) - {"batch"}
        export_kwargs = {k: v for k, v in args.items() if k in export_args}
        return benchmark(
            model=self,
            data=data,  # 如果未传递 'data' 参数，则为默认数据集设置 data=None
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            **export_kwargs,
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """将模型导出为适合部署的不同格式。

        此方法促进将模型导出为各种格式（例如 ONNX、TorchScript）以用于部署。
        它使用 'Exporter' 类进行导出过程，结合模型特定的覆盖、方法默认值和提供的任何其他参数。

        参数:
            **kwargs (Any): 用于导出配置的任意关键字参数。常用选项包括:
                - format (str): 导出格式（例如 'onnx'、'engine'、'coreml'）。
                - half (bool): 以半精度导出模型。
                - int8 (bool): 以 int8 精度导出模型。
                - device (str): 运行导出的设备。
                - workspace (int): TensorRT 引擎的最大内存工作空间大小。
                - nms (bool): 向模型添加非极大值抑制（NMS）模块。
                - simplify (bool): 简化 ONNX 模型。

        返回:
            (str): 导出模型文件的路径。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。
            ValueError: 如果指定了不受支持的导出格式。
            RuntimeError: 如果导出过程由于错误而失败。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # 重置以避免多 GPU 错误
            "verbose": False,
        }  # 方法默认值
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # 右侧的参数优先级最高
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """使用指定的数据集和训练配置训练模型。

        此方法通过一系列可自定义的设置促进模型训练。它支持使用自定义训练器或默认训练方法
        进行训练。该方法处理从检查点恢复训练、与 Ultralytics HUB 集成以及训练后更新模型
        和配置等场景。

        使用 Ultralytics HUB 时，如果会话已加载模型，该方法会优先使用 HUB 训练参数，
        并在提供本地参数时发出警告。它检查 pip 更新并结合默认配置、方法特定默认值和
        用户提供的参数来配置训练过程。

        参数:
            trainer (BaseTrainer, optional): 用于模型训练的自定义训练器实例。如果为 None，则使用默认值。
            **kwargs (Any): 用于训练配置的任意关键字参数。常用选项包括:
                - data (str): 数据集配置文件的路径。
                - epochs (int): 训练轮数。
                - batch (int): 训练批次大小。
                - imgsz (int): 输入图像大小。
                - device (str): 运行训练的设备（例如 'cuda'、'cpu'）。
                - workers (int): 用于数据加载的工作线程数。
                - optimizer (str): 用于训练的优化器。
                - lr0 (float): 初始学习率。
                - patience (int): 在没有可观察的改进时等待的轮数，用于提前停止训练。
                - augmentations (list[Callable]): 训练期间要应用的增强函数列表。

        返回:
            (dict | None): 如果可用且训练成功，则返回训练指标；否则返回 None。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # 带有已加载模型的 Ultralytics HUB 会话
            if any(kwargs):
                LOGGER.warning("using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # 覆盖 kwargs

        checks.check_pip_update_available()

        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            self.load(kwargs["pretrained"])  # 如果提供，则加载预训练权重
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # 注意: 处理 'cfg' 包含 'data' 的情况
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # 方法默认值
        args = {**overrides, **custom, **kwargs, "mode": "train", "session": self.session}  # 右侧的参数优先
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # 仅在不恢复时手动设置模型
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.train()
        # 训练后更新模型和配置
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: DDP 不返回指标
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """对模型进行超参数调优，可选择使用 Ray Tune。

        此方法支持两种超参数调优模式: 使用 Ray Tune 或自定义调优方法。启用 Ray Tune 时，
        它利用 ultralytics.utils.tuner 模块中的 'run_ray_tune' 函数。否则，它使用内部的
        'Tuner' 类进行调优。该方法结合默认、覆盖和自定义参数来配置调优过程。

        参数:
            use_ray (bool): 是否使用 Ray Tune 进行超参数调优。如果为 False，则使用内部调优方法。
            iterations (int): 要执行的调优迭代次数。
            *args (Any): 要传递给调优器的其他位置参数。
            **kwargs (Any): 用于调优配置的其他关键字参数。这些与模型覆盖和默认值结合以配置调优过程。

        返回:
            (dict): 超参数搜索的结果，包括最佳参数和性能指标。

        异常:
            TypeError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # 使用 Ray Tune 进行更高级的超参数搜索
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # 方法默认值
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # 右侧的参数优先级最高
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> Model:
        """将函数应用于不是参数或注册缓冲区的模型张量。

        此方法通过额外重置预测器并更新模型覆盖中的设备来扩展父类的 _apply 方法的功能。
        它通常用于将模型移动到不同设备或更改其精度等操作。

        参数:
            fn (Callable): 要应用于模型张量的函数。这通常是 to()、cpu()、cuda()、half() 或 float() 等方法。

        返回:
            (Model): 应用了函数并更新了属性的模型实例。

        异常:
            AssertionError: 如果模型不是 PyTorch 模型。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # 将模型移动到 GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None  # 重置预测器，因为设备可能已更改
        self.overrides["device"] = self.device  # 之前是 str(self.device)，即 device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> dict[int, str]:
        """检索与已加载模型关联的类名。

        此属性在模型中定义类名时返回类名。它使用 ultralytics.nn.autobackend 模块中的
        'check_class_names' 函数检查类名的有效性。如果预测器未初始化，它会在检索名称之前设置它。

        返回:
            (dict[int, str]): 与模型关联的类名字典，其中键是类索引，值是相应的类名。

        异常:
            AttributeError: 如果模型或预测器没有 'names' 属性。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # 导出格式在调用 predict() 之前不会定义预测器
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)  # 不要修改 self.predictor.model 参数
            return predictor.model.names
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """获取模型参数所分配的设备。

        此属性确定模型参数当前存储的设备（CPU 或 GPU）。它仅适用于 torch.nn.Module 实例的模型。

        返回:
            (torch.device): 模型的设备（CPU/GPU）。

        异常:
            AttributeError: 如果模型不是 torch.nn.Module 实例。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # 如果 CUDA 可用
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """检索应用于已加载模型输入数据的转换。

        此属性在模型中定义转换时返回转换。转换通常包括预处理步骤，如调整大小、归一化和
        数据增强，这些步骤在将输入数据馈送到模型之前应用。

        返回:
            (object | None): 模型的转换对象（如果可用），否则为 None。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """为指定事件添加回调函数。

        此方法允许注册在训练或推理等模型操作期间在特定事件上触发的自定义回调函数。
        回调提供了一种在模型生命周期的各个阶段扩展和自定义模型行为的方法。

        参数:
            event (str): 要附加回调的事件名称。必须是 Ultralytics 框架识别的有效事件名称。
            func (Callable): 要注册的回调函数。指定事件发生时将调用此函数。

        异常:
            ValueError: 如果事件名称未被识别或无效。

        示例:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """清除为指定事件注册的所有回调函数。

        此方法删除与给定事件关联的所有自定义和默认回调函数。它将指定事件的回调列表
        重置为空列表，从而有效地删除该事件的所有已注册回调。

        参数:
            event (str): 要清除回调的事件名称。这应该是 Ultralytics 回调系统识别的有效事件名称。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # 'on_train_start' 的所有回调现在都已删除

        注意:
            - 此方法影响用户添加的自定义回调和 Ultralytics 框架提供的默认回调。
            - 调用此方法后，在添加新回调之前，不会为指定事件执行任何回调。
            - 谨慎使用，因为它会删除所有回调，包括某些操作正常运行可能需要的基本回调。
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """将所有回调重置为其默认函数。

        此方法为所有事件恢复默认回调函数，删除之前添加的任何自定义回调。
        它遍历所有默认回调事件并将当前回调替换为默认回调。

        默认回调在 'callbacks.default_callbacks' 字典中定义，该字典包含模型生命周期中
        各种事件的预定义函数，如 on_train_start、on_epoch_end 等。

        当您想在进行自定义修改后恢复到原始回调集时，此方法很有用，
        可确保不同运行或实验之间的行为一致。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # 所有回调现在都已重置为其默认函数
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        """加载 PyTorch 模型检查点时重置特定参数。

        此方法过滤输入参数字典，仅保留被认为对模型加载很重要的特定键集。
        它用于确保从检查点加载模型时仅保留相关参数，丢弃任何不必要或可能冲突的设置。

        参数:
            args (dict): 包含各种模型参数和设置的字典。

        返回:
            (dict): 仅包含输入参数中指定包含键的新字典。

        示例:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # 加载 PyTorch 模型时仅记住这些参数
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """如果对象没有请求的属性，则引发错误。"""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """根据模型任务智能加载适当的模块。

        此方法根据模型的当前任务和提供的键动态选择并返回正确的模块（模型、训练器、验证器或预测器）。
        它使用 task_map 字典来确定要为特定任务加载的适当模块。

        参数:
            key (str): 要加载的模块类型。必须是 'model'、'trainer'、'validator' 或 'predictor' 之一。

        返回:
            (object): 与指定键和当前任务对应的已加载模块类。

        异常:
            NotImplementedError: 如果当前任务不支持指定的键。

        示例:
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # 获取函数名称
            raise NotImplementedError(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.") from e

    @property
    def task_map(self) -> dict:
        """提供从模型任务到不同模式的相应类的映射。

        此属性方法返回一个字典，该字典将每个支持的任务（例如 detect、segment、classify）
        映射到嵌套字典。嵌套字典包含不同操作模式（model、trainer、validator、predictor）
        到其各自类实现的映射。

        该映射允许根据模型的任务和所需的操作模式动态加载适当的类。
        这有助于在 Ultralytics 框架内处理各种任务和模式的灵活且可扩展的架构。

        返回:
            (dict[str, dict[str, Any]]): 将任务名称映射到嵌套字典的字典。每个嵌套字典包含
                'model'、'trainer'、'validator' 和 'predictor' 键到该任务各自类实现的映射。

        示例:
            >>> model = Model("yolo11n.pt")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """将模型设置为评估模式。

        此方法将模型的模式更改为评估模式，这会影响 dropout 和 batch normalization 等层，
        它们在训练和评估期间的行为不同。在评估模式下，这些层使用运行统计信息而不是计算批次统计信息，
        并且 dropout 层被禁用。

        返回:
            (Model): 设置了评估模式的模型实例。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> model.eval()
            >>> # 模型现在处于评估模式以进行推理
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """允许通过 Model 类直接访问模型属性。

        此方法提供了一种直接通过 Model 类实例访问底层模型属性的方法。
        它首先检查请求的属性是否为 'model'，在这种情况下，它从模块字典返回模型。
        否则，它将属性查找委托给底层模型。

        参数:
            name (str): 要检索的属性名称。

        返回:
            (Any): 请求的属性值。

        异常:
            AttributeError: 如果请求的属性在模型中不存在。

        示例:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)  # 访问 model.stride 属性
            >>> print(model.names)  # 访问 model.names 属性
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
