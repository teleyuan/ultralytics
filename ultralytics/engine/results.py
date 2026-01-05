# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
推理结果处理模块

该模块提供了 Ultralytics YOLO 模型推理结果的封装和处理类，包括边界框、掩码、关键点等。
主要用于统一处理各种检测任务的输出结果，提供便捷的结果访问和可视化方法。

主要类:
    - BaseTensor: 张量基类，提供设备管理和类型转换
    - Results: 推理结果主类，封装图像、预测和元数据
    - Boxes: 边界框类，处理检测框数据
    - Masks: 掩码类，处理分割掩码
    - Keypoints: 关键点类，处理姿态估计关键点
    - Probs: 概率类，处理分类任务的类别概率
    - OBB: 有向边界框类，处理旋转目标检测

使用方法: 参见 https://docs.ultralytics.com/modes/predict/
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import deepcopy  # 深拷贝对象
from functools import lru_cache  # LRU 缓存装饰器
from pathlib import Path  # 跨平台路径操作
from typing import Any  # 类型提示

import numpy as np  # 数组和数值计算
import torch  # PyTorch 深度学习框架

# 数据增强和工具导入
from ultralytics.data.augment import LetterBox  # 图像填充变换
from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, ops  # 工具类和操作
from ultralytics.utils.plotting import Annotator, colors, save_one_box  # 可视化工具


class BaseTensor(SimpleClass):
    """基础张量类,提供额外的方法用于简便操作和设备处理。

    该类为类张量对象提供基础,具备设备管理能力,支持 PyTorch 张量和 NumPy 数组。
    包含在设备间移动数据和在张量类型间转换的方法。

    Attributes:
        data (torch.Tensor | np.ndarray): 预测数据,如边界框、掩码或关键点。
        orig_shape (tuple[int, int]): 图像的原始形状,通常为 (高度, 宽度) 格式。

    Methods:
        cpu: 返回存储在 CPU 内存中的张量副本。
        numpy: 返回张量的 numpy 数组副本。
        cuda: 将张量移至 GPU 内存,必要时返回新实例。
        to: 返回具有指定设备和数据类型的张量副本。

    Examples:
        >>> import torch
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape = (720, 1280)
        >>> base_tensor = BaseTensor(data, orig_shape)
        >>> cpu_tensor = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor = base_tensor.cuda()
    """

    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """初始化 BaseTensor，包含预测数据和原始图像形状

        Args:
            data (torch.Tensor | np.ndarray): 预测数据，如边界框、掩码或关键点
            orig_shape (tuple[int, int]): 原始图像形状，格式为 (高度, 宽度)
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data  # 存储预测数据（张量或数组）
        self.orig_shape = orig_shape  # 存储原始图像尺寸

    @property
    def shape(self) -> tuple[int, ...]:
        """返回底层数据张量的形状。

        Returns:
            (tuple[int, ...]): 数据张量的形状。

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """返回存储在 CPU 内存中的张量副本。

        Returns:
            (BaseTensor): 数据张量已移至 CPU 内存的新 BaseTensor 对象。

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensor)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """返回数据已转换为 NumPy 数组的对象副本。

        Returns:
            (BaseTensor): `data` 为 NumPy 数组的新实例。

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseTensor(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """将张量移至 GPU 内存。

        Returns:
            (BaseTensor): 数据已移至 GPU 内存的新 BaseTensor 实例。

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """返回具有指定设备和数据类型的张量副本。

        Args:
            *args (Any): 要传递给 torch.Tensor.to() 的可变长度参数列表。
            **kwargs (Any): 要传递给 torch.Tensor.to() 的任意关键字参数。

        Returns:
            (BaseTensor): 数据已移至指定设备和/或数据类型的新 BaseTensor 实例。

        Examples:
            >>> base_tensor = BaseTensor(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """返回底层数据张量的长度。

        Returns:
            (int): 数据张量第一维度的元素数量。

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """返回包含数据张量指定索引元素的新 BaseTensor 实例。

        Args:
            idx (int | list[int] | torch.Tensor): 要从数据张量中选择的索引或索引集。

        Returns:
            (BaseTensor): 包含索引数据的新 BaseTensor 实例。

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # 选择第一行
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass, DataExportMixin):
    """用于存储和操作推理结果的类。

    该类为处理各种 Ultralytics 模型的推理结果提供全面的功能,
    包括检测、分割、分类和姿态估计。支持可视化、数据导出和各种坐标转换。

    Attributes:
        orig_img (np.ndarray): 作为 numpy 数组的原始图像。
        orig_shape (tuple[int, int]): 原始图像形状,格式为 (高度, 宽度)。
        boxes (Boxes | None): 检测到的边界框。
        masks (Masks | None): 分割掩码。
        probs (Probs | None): 分类概率。
        keypoints (Keypoints | None): 检测到的关键点。
        obb (OBB | None): 有向边界框。
        speed (dict): 包含推理速度信息的字典。
        names (dict): 将类别索引映射到类别名称的字典。
        path (str): 输入图像文件的路径。
        save_dir (str | None): 保存结果的目录。

    Methods:
        update: 使用新的检测数据更新 Results 对象。
        cpu: 返回所有张量已移至 CPU 内存的 Results 对象副本。
        numpy: 将 Results 对象中的所有张量转换为 numpy 数组。
        cuda: 将 Results 对象中的所有张量移至 GPU 内存。
        to: 将所有张量移至指定设备和数据类型。
        new: 创建具有相同图像、路径、名称和速度属性的新 Results 对象。
        plot: 在输入 BGR 图像上绘制检测结果。
        show: 显示带有注释推理结果的图像。
        save: 将带注释的推理结果图像保存到文件。
        verbose: 返回结果中每个任务的日志字符串。
        save_txt: 将检测结果保存到文本文件。
        save_crop: 将裁剪的检测图像保存到指定目录。
        summary: 将推理结果转换为摘要字典。
        to_df: 将检测结果转换为 Polars DataFrame。
        to_json: 将检测结果转换为 JSON 格式。
        to_csv: 将检测结果转换为 CSV 格式。

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # 获取第一个结果
        >>> boxes = result.boxes  # 获取第一个结果的边界框
        >>> masks = result.masks  # 获取第一个结果的掩码
        >>> for result in results:
        >>>     result.plot()  # 绘制检测结果
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        """初始化 Results 类,用于存储和操作推理结果。

        Args:
            orig_img (np.ndarray): 作为 numpy 数组的原始图像。
            path (str): 图像文件的路径。
            names (dict): 类别名称的字典。
            boxes (torch.Tensor | None): 每个检测的边界框坐标的二维张量。
            masks (torch.Tensor | None): 检测掩码的三维张量,其中每个掩码都是一个二值图像。
            probs (torch.Tensor | None): 分类任务中每个类别的概率的一维张量。
            keypoints (torch.Tensor | None): 每个检测的关键点坐标的二维张量。
            obb (torch.Tensor | None): 每个检测的有向边界框坐标的二维张量。
            speed (dict | None): 包含预处理、推理和后处理速度的字典 (毫秒/图像)。

        Notes:
            对于默认姿态模型,人体姿态估计的关键点索引为:
            0: 鼻子, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
            5: 左肩, 6: 右肩, 7: 左肘, 8: 右肘
            9: 左腕, 10: 右腕, 11: 左髋, 12: 右髋
            13: 左膝, 14: 右膝, 15: 左踝, 16: 右踝
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """返回推理结果特定索引的 Results 对象。

        Args:
            idx (int | slice): 要从 Results 对象检索的索引或切片。

        Returns:
            (Results): 包含指定推理结果子集的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> single_result = results[0]  # 获取第一个结果
            >>> subset_results = results[1:4]  # 获取结果切片
        """
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        """返回 Results 对象中的检测数量。

        Returns:
            (int): 检测数量,由 (masks, probs, keypoints 或 obb) 中第一个非空属性的长度确定。

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 4))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(
        self,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
    ):
        """使用新的检测数据更新 Results 对象。

        该方法允许更新 Results 对象的边界框、掩码、概率和有向边界框 (OBB)。
        它确保边界框被裁剪到原始图像形状。

        Args:
            boxes (torch.Tensor | None): 形状为 (N, 6) 的张量,包含边界框坐标和置信度分数。
                格式为 (x1, y1, x2, y2, conf, class)。
            masks (torch.Tensor | None): 形状为 (N, H, W) 的张量,包含分割掩码。
            probs (torch.Tensor | None): 形状为 (num_classes,) 的张量,包含类别概率。
            obb (torch.Tensor | None): 形状为 (N, 5) 的张量,包含有向边界框坐标。
            keypoints (torch.Tensor | None): 形状为 (N, 17, 3) 的张量,包含关键点。

        Examples:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """对所有非空属性应用函数,并返回具有修改属性的新 Results 对象。

        该方法由 .to(), .cuda(), .cpu() 等方法内部调用。

        Args:
            fn (str): 要应用的函数名称。
            *args (Any): 要传递给函数的可变长度参数列表。
            **kwargs (Any): 要传递给函数的任意关键字参数。

        Returns:
            (Results): 属性已由应用函数修改的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """返回所有张量已移至 CPU 内存的 Results 对象副本。

        该方法创建一个新的 Results 对象,其所有张量属性 (boxes, masks, probs, keypoints, obb)
        已转移到 CPU 内存。用于将数据从 GPU 移至 CPU 以进行进一步处理或保存。

        Returns:
            (Results): 所有张量属性都在 CPU 内存上的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")  # 执行推理
            >>> cpu_result = results[0].cpu()  # 将第一个结果移至 CPU
            >>> print(cpu_result.boxes.device)  # 输出: cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """将 Results 对象中的所有张量转换为 numpy 数组。

        Returns:
            (Results): 所有张量已转换为 numpy 数组的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            该方法创建一个新的 Results 对象,原对象保持不变。
            用于与基于 numpy 的库互操作或需要基于 CPU 的操作时。
        """
        return self._apply("numpy")

    def cuda(self):
        """将 Results 对象中的所有张量移至 GPU 内存。

        Returns:
            (Results): 所有张量已移至 CUDA 设备的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # 将第一个结果移至 GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # 将每个结果移至 GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """将 Results 对象中的所有张量移至指定设备和数据类型。

        Args:
            *args (Any): 要传递给 torch.Tensor.to() 的可变长度参数列表。
            **kwargs (Any): 要传递给 torch.Tensor.to() 的任意关键字参数。

        Returns:
            (Results): 所有张量已移至指定设备和数据类型的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # 将第一个结果移至 GPU
            >>> result_cpu = results[0].to("cpu")  # 将第一个结果移至 CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # 将第一个结果转换为半精度
        """
        return self._apply("to", *args, **kwargs)

    def new(self):
        """创建具有相同图像、路径、名称和速度属性的新 Results 对象。

        Returns:
            (Results): 从原实例复制属性的新 Results 对象。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | None = None,
        im_gpu: torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """在输入 BGR 图像上绘制检测结果。

        Args:
            conf (bool): 是否绘制检测置信度分数。
            line_width (float | None): 边界框的线宽。如果为 None,则根据图像大小缩放。
            font_size (float | None): 文本的字体大小。如果为 None,则根据图像大小缩放。
            font (str): 用于文本的字体。
            pil (bool): 是否将图像作为 PIL Image 返回。
            img (np.ndarray | None): 要绘制的图像。如果为 None,则使用原始图像。
            im_gpu (torch.Tensor | None): GPU 上的归一化图像,用于更快的掩码绘制。
            kpt_radius (int): 绘制关键点的半径。
            kpt_line (bool): 是否绘制连接关键点的线。
            labels (bool): 是否绘制边界框的标签。
            boxes (bool): 是否绘制边界框。
            masks (bool): 是否绘制掩码。
            probs (bool): 是否绘制分类概率。
            show (bool): 是否显示带注释的图像。
            save (bool): 是否保存带注释的图像。
            filename (str | None): 如果 save 为 True,保存图像的文件名。
            color_mode (str): 指定颜色模式,例如 'instance' 或 'class'。
            txt_color (tuple[int, int, int]): 分类输出的文本颜色,BGR 格式。

        Returns:
            (np.ndarray | PIL.Image.Image): 带注释的图像,作为 NumPy 数组 (BGR) 或 PIL 图像 (RGB)(如果 `pil=True`)。

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            >>>     im = result.plot()
            >>>     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, int(d.id.item()) if d.is_track else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA box

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.result(pil)

    def show(self, *args, **kwargs):
        """显示带有注释推理结果的图像。

        该方法在原始图像上绘制检测结果并显示它。这是直接可视化模型预测的便捷方法。

        Args:
            *args (Any): 要传递给 `plot()` 方法的可变长度参数列表。
            **kwargs (Any): 要传递给 `plot()` 方法的任意关键字参数。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # 显示第一个结果
            >>> for result in results:
            >>>     result.show()  # 显示所有结果
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename: str | None = None, *args, **kwargs) -> str:
        """将带注释的推理结果图像保存到文件。

        该方法在原始图像上绘制检测结果并将带注释的图像保存到文件。
        它利用 `plot` 方法生成带注释的图像,然后将其保存到指定的文件名。

        Args:
            filename (str | Path | None): 保存带注释图像的文件名。如果为 None,
                则根据原始图像路径生成默认文件名。
            *args (Any): 要传递给 `plot` 方法的可变长度参数列表。
            **kwargs (Any): 要传递给 `plot` 方法的任意关键字参数。

        Returns:
            (str): 图像保存的文件名。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save("annotated_image.jpg")
            >>> # 或使用自定义绘图参数
            >>> for result in results:
            >>>     result.save("annotated_image.jpg", conf=False, line_width=2)
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self) -> str:
        """返回结果中每个任务的日志字符串,详细说明检测和分类结果。

        该方法生成一个人类可读的字符串,总结检测和分类结果。
        它包括每个类别的检测数量和分类任务的最高概率。

        Returns:
            (str): 包含结果摘要的格式化字符串。对于检测任务,它包括每个类别的检测数量。
                对于分类任务,它包括前 5 个类别概率。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - 如果没有检测,该方法对检测任务返回 "(no detections), "。
            - 对于分类任务,它返回前 5 个类别概率及其对应的类别名称。
            - 返回的字符串以逗号分隔,并以逗号和空格结尾。
        """
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return "" if self.probs is not None else "(no detections), "
        if self.probs is not None:
            return f"{', '.join(f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5)}, "
        if boxes:
            counts = boxes.cls.int().bincount()
            return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """将检测结果保存到文本文件。

        Args:
            txt_file (str | Path): 输出文本文件的路径。
            save_conf (bool): 是否在输出中包含置信度分数。

        Returns:
            (str): 保存的文本文件的路径。

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_txt("output.txt")

        Notes:
            - 文件将包含每个检测或分类的一行,具有以下结构:
              - 对于检测: `class confidence x_center y_center width height`
              - 对于分类: `confidence class_name`
              - 对于掩码和关键点,特定格式会相应变化。
            - 如果输出目录不存在,该函数将创建它。
            - 如果 save_conf 为 False,置信度分数将从输出中排除。
            - 不会覆盖文件的现有内容;新结果将被追加。
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)

    def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
        """将裁剪的检测图像保存到指定目录。

        该方法将检测到的对象的裁剪图像保存到指定目录。每个裁剪图像都保存在
        以对象类别命名的子目录中,文件名基于输入的 file_name。

        Args:
            save_dir (str | Path): 保存裁剪图像的目录路径。
            file_name (str | Path): 保存的裁剪图像的基本文件名。

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_crop(save_dir="path/to/crops", file_name="detection")

        Notes:
            - 该方法不支持分类或有向边界框 (OBB) 任务。
            - 裁剪图像保存为 'save_dir/class_name/file_name.jpg'。
            - 如果子目录不存在,该方法将创建必要的子目录。
            - 在裁剪前复制原始图像以避免修改原始图像。
        """
        if self.probs is not None:
            LOGGER.warning("Classify task does not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task does not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]:
        """将推理结果转换为摘要字典,可选择对边界框坐标进行归一化。

        该方法创建一个检测字典列表,每个字典包含有关单个检测或分类结果的信息。
        对于分类任务,它返回最高类别及其置信度。对于检测任务,它包括类别信息、
        边界框坐标,以及可选的掩码片段和关键点。

        Args:
            normalize (bool): 是否根据图像尺寸归一化边界框坐标。
            decimals (int): 输出值要舍入到的小数位数。

        Returns:
            (list[dict[str, Any]]): 字典列表,每个字典包含单个检测或分类结果的摘要信息。
                每个字典的结构根据任务类型(分类或检测)和可用信息(边界框、掩码、关键点)而变化。

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            >>>     summary = result.summary()
            >>>     print(summary)
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results


class Boxes(BaseTensor):
    """用于管理和操作检测框的类。

    该类为处理检测框提供全面的功能,包括其坐标、置信度分数、类别标签和可选的跟踪 ID。
    它支持各种边界框格式,并提供在不同坐标系之间轻松操作和转换的方法。

    Attributes:
        data (torch.Tensor | np.ndarray): 包含检测框和关联数据的原始张量。
        orig_shape (tuple[int, int]): 原始图像尺寸 (高度, 宽度)。
        is_track (bool): 指示边界框数据中是否包含跟踪 ID。
        xyxy (torch.Tensor | np.ndarray): [x1, y1, x2, y2] 格式的边界框。
        conf (torch.Tensor | np.ndarray): 每个边界框的置信度分数。
        cls (torch.Tensor | np.ndarray): 每个边界框的类别标签。
        id (torch.Tensor | None): 每个边界框的跟踪 ID(如果可用)。
        xywh (torch.Tensor | np.ndarray): [x, y, width, height] 格式的边界框。
        xyxyn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 [x1, y1, x2, y2] 边界框。
        xywhn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 [x, y, width, height] 边界框。

    Methods:
        cpu: 返回所有张量在 CPU 内存上的对象副本。
        numpy: 返回所有张量为 numpy 数组的对象副本。
        cuda: 返回所有张量在 GPU 内存上的对象副本。
        to: 返回张量在指定设备和数据类型上的对象副本。

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # 高度, 宽度
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测框数据和原始图像形状初始化 Boxes 类。

        该类管理检测框,提供对边界框坐标、置信度分数、类别标识符和可选跟踪 ID 的轻松访问和操作。
        它支持边界框坐标的多种格式,包括绝对和归一化形式。

        Args:
            boxes (torch.Tensor | np.ndarray): 形状为 (num_boxes, 6) 或 (num_boxes, 7) 的张量或 numpy 数组。
                列应包含 [x1, y1, x2, y2, (可选) track_id, confidence, class]。
            orig_shape (tuple[int, int]): 原始图像形状,格式为 (高度, 宽度)。用于归一化。
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """返回 [x1, y1, x2, y2] 格式的边界框。

        Returns:
            (torch.Tensor | np.ndarray): 形状为 (n, 4) 的张量或 numpy 数组,包含 [x1, y1, x2, y2] 格式的
                边界框坐标,其中 n 是边界框数量。

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """返回每个检测框的置信度分数。

        Returns:
            (torch.Tensor | np.ndarray): 包含每个检测的置信度分数的一维张量或数组,
                形状为 (N,),其中 N 是检测数量。

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """返回表示每个边界框类别预测的类别 ID 张量。

        Returns:
            (torch.Tensor | np.ndarray): 包含每个检测框的类别 ID 的张量或 numpy 数组。
                形状为 (N,),其中 N 是边界框数量。

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """返回每个检测框的跟踪 ID(如果可用)。

        Returns:
            (torch.Tensor | None): 如果启用跟踪,则包含每个边界框的跟踪 ID 的张量,否则为 None。
                形状为 (N,),其中 N 是边界框数量。

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - 仅当启用跟踪时(即 `is_track` 为 True 时)此属性才可用。
            - 跟踪 ID 通常用于在视频分析中关联多个帧之间的检测。
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self) -> torch.Tensor | np.ndarray:
        """将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, width, height] 格式。

        Returns:
            (torch.Tensor | np.ndarray): [x_center, y_center, width, height] 格式的边界框,其中 x_center, y_center
                是边界框中心点的坐标,width, height 是边界框的尺寸,返回张量的形状为 (N, 4),其中 N 是边界框数量。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[125.0000,  75.0000,  50.0000,  50.0000],
                    [250.0000, 200.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """返回相对于原始图像大小归一化的边界框坐标。

        该属性计算并返回 [x1, y1, x2, y2] 格式的边界框坐标,
        基于原始图像尺寸归一化到 [0, 1] 范围。

        Returns:
            (torch.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框坐标,其中 N 是边界框数量。
                每行包含归一化到 [0, 1] 的 [x1, y1, x2, y2] 值。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """返回 [x, y, width, height] 格式的归一化边界框。

        该属性计算并返回 [x_center, y_center, width, height] 格式的归一化边界框坐标,
        其中所有值都相对于原始图像尺寸。

        Returns:
            (torch.Tensor | np.ndarray): 形状为 (N, 4) 的归一化边界框,其中 N 是边界框数量。
                每行包含基于原始图像尺寸归一化到 [0, 1] 的 [x_center, y_center, width, height] 值。

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """用于存储和操作检测掩码的类。

    该类扩展 BaseTensor 并提供处理分割掩码的功能,包括在像素坐标和归一化坐标之间转换的方法。

    Attributes:
        data (torch.Tensor | np.ndarray): 包含掩码数据的原始张量或数组。
        orig_shape (tuple): 原始图像形状,格式为 (高度, 宽度)。
        xy (list[np.ndarray]): 像素坐标中的片段列表。
        xyn (list[np.ndarray]): 归一化片段的列表。

    Methods:
        cpu: 返回掩码张量在 CPU 内存上的 Masks 对象副本。
        numpy: 返回掩码张量为 numpy 数组的 Masks 对象副本。
        cuda: 返回掩码张量在 GPU 内存上的 Masks 对象副本。
        to: 返回掩码张量在指定设备和数据类型上的 Masks 对象副本。

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测掩码数据和原始图像形状初始化 Masks 类。

        Args:
            masks (torch.Tensor | np.ndarray): 形状为 (num_masks, height, width) 的检测掩码。
            orig_shape (tuple): 原始图像形状,格式为 (高度, 宽度)。用于归一化。
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> list[np.ndarray]:
        """返回分割掩码的归一化 xy 坐标。

        该属性计算并缓存分割掩码的归一化 xy 坐标。坐标相对于原始图像形状进行归一化。

        Returns:
            (list[np.ndarray]): numpy 数组列表,其中每个数组包含单个分割掩码的归一化 xy 坐标。
                每个数组的形状为 (N, 2),其中 N 是掩码轮廓中的点数。

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # 第一个掩码的归一化坐标
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> list[np.ndarray]:
        """返回掩码张量中每个片段的 [x, y] 像素坐标。

        该属性计算并返回 Masks 对象中每个分割掩码的像素坐标列表。
        坐标被缩放以匹配原始图像尺寸。

        Returns:
            (list[np.ndarray]): numpy 数组列表,其中每个数组包含单个分割掩码的 [x, y] 像素坐标。
                每个数组的形状为 (N, 2),其中 N 是片段中的点数。

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # 掩码数量
            >>> print(xy_coords[0].shape)  # 第一个掩码坐标的形状
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)
        ]


class Keypoints(BaseTensor):
    """用于存储和操作检测关键点的类。

    该类封装了处理关键点数据的功能,包括坐标操作、归一化和置信度值。
    它支持带有可选可见性信息的关键点检测结果。

    Attributes:
        data (torch.Tensor): 包含关键点数据的原始张量。
        orig_shape (tuple[int, int]): 原始图像尺寸 (高度, 宽度)。
        has_visible (bool): 指示关键点是否有可见性信息。
        xy (torch.Tensor): [x, y] 格式的关键点坐标。
        xyn (torch.Tensor): [x, y] 格式的归一化关键点坐标,相对于 orig_shape。
        conf (torch.Tensor): 每个关键点的置信度值(如果可用)。

    Methods:
        cpu: 返回关键点张量在 CPU 内存上的副本。
        numpy: 返回关键点张量的 numpy 数组副本。
        cuda: 返回关键点张量在 GPU 内存上的副本。
        to: 返回具有指定设备和数据类型的关键点张量副本。

    Examples:
        >>> import torch
        >>> from ultralytics.engine.results import Keypoints
        >>> keypoints_data = torch.rand(1, 17, 3)  # 1 个检测, 17 个关键点, (x, y, conf)
        >>> orig_shape = (480, 640)  # 原始图像形状 (高度, 宽度)
        >>> keypoints = Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape)  # 访问 xy 坐标
        >>> print(keypoints.conf)  # 访问置信度值
        >>> keypoints_cpu = keypoints.cpu()  # 将关键点移至 CPU
    """

    def __init__(self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用检测关键点和原始图像尺寸初始化 Keypoints 对象。

        该方法处理输入关键点张量,处理二维和三维格式。对于三维张量 (x, y, confidence),
        它通过将低置信度关键点的坐标设置为零来屏蔽它们。

        Args:
            keypoints (torch.Tensor): 包含关键点数据的张量。形状可以是:
                - (num_objects, num_keypoints, 2) 仅用于 x, y 坐标
                - (num_objects, num_keypoints, 3) 用于 x, y 坐标和置信度分数
            orig_shape (tuple[int, int]): 原始图像尺寸 (高度, 宽度)。
        """
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self) -> torch.Tensor | np.ndarray:
        """返回关键点的 x, y 坐标。

        Returns:
            (torch.Tensor): 包含关键点 x, y 坐标的张量,形状为 (N, K, 2),其中 N 是检测数量,
                K 是每个检测的关键点数量。

        Examples:
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # 第一个检测的关键点 x, y 坐标

        Notes:
            - 返回的坐标是相对于原始图像尺寸的像素单位。
            - 如果关键点使用置信度值初始化,则仅返回置信度 >= 0.5 的关键点。
            - 此属性使用 LRU 缓存以提高重复访问的性能。
        """
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self) -> torch.Tensor | np.ndarray:
        """返回相对于原始图像大小的关键点归一化坐标 (x, y)。

        Returns:
            (torch.Tensor | np.ndarray): 形状为 (N, K, 2) 的张量或数组,包含归一化关键点坐标,
                其中 N 是实例数量,K 是关键点数量,最后一个维度包含 [0, 1] 范围内的 [x, y] 值。

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self) -> torch.Tensor | np.ndarray | None:
        """返回每个关键点的置信度值。

        Returns:
            (torch.Tensor | None): 包含每个关键点置信度分数的张量(如果可用),否则为 None。
                对于批量数据,形状为 (num_detections, num_keypoints);对于单个检测,形状为 (num_keypoints,)。

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 个检测, 17 个关键点
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """用于存储和操作分类概率的类。

    该类扩展 BaseTensor 并提供用于访问和操作分类概率的方法,包括 top-1 和 top-5 预测。

    Attributes:
        data (torch.Tensor | np.ndarray): 包含分类概率的原始张量或数组。
        orig_shape (tuple | None): 原始图像形状,格式为 (高度, 宽度)。此类中未使用。
        top1 (int): 具有最高概率的类别索引。
        top5 (list[int]): 按概率排序的前 5 个类别的索引。
        top1conf (torch.Tensor | np.ndarray): top 1 类别的置信度分数。
        top5conf (torch.Tensor | np.ndarray): top 5 类别的置信度分数。

    Methods:
        cpu: 返回概率张量在 CPU 内存上的副本。
        numpy: 返回概率张量的 numpy 数组副本。
        cuda: 返回概率张量在 GPU 内存上的副本。
        to: 返回具有指定设备和数据类型的概率张量副本。

    Examples:
        >>> probs = torch.tensor([0.1, 0.3, 0.6])
        >>> p = Probs(probs)
        >>> print(p.top1)
        2
        >>> print(p.top5)
        [2, 1, 0]
        >>> print(p.top1conf)
        tensor(0.6000)
        >>> print(p.top5conf)
        tensor([0.6000, 0.3000, 0.1000])
    """

    def __init__(self, probs: torch.Tensor | np.ndarray, orig_shape: tuple[int, int] | None = None) -> None:
        """使用分类概率初始化 Probs 类。

        该类存储和管理分类概率,提供对最高预测及其置信度的轻松访问。

        Args:
            probs (torch.Tensor | np.ndarray): 分类概率的一维张量或数组。
            orig_shape (tuple | None): 原始图像形状,格式为 (高度, 宽度)。
                此类中未使用,但为与其他结果类保持一致而保留。
        """
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self) -> int:
        """返回具有最高概率的类别索引。

        Returns:
            (int): 具有最高概率的类别索引。

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.3, 0.6]))
            >>> probs.top1
            2
        """
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self) -> list[int]:
        """返回前 5 个类别概率的索引。

        Returns:
            (list[int]): 包含前 5 个类别概率索引的列表,按降序排列。

        Examples:
            >>> probs = Probs(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
            >>> print(probs.top5)
            [4, 3, 2, 1, 0]
        """
        return (-self.data).argsort(0)[:5].tolist()  # 这种方式适用于 torch 和 numpy

    @property
    @lru_cache(maxsize=1)
    def top1conf(self) -> torch.Tensor | np.ndarray:
        """返回最高概率类别的置信度分数。

        该属性从分类结果中检索具有最高预测概率的类别的置信度分数(概率)。

        Returns:
            (torch.Tensor | np.ndarray): 包含 top 1 类别置信度分数的张量。

        Examples:
            >>> results = model("image.jpg")  # 对图像进行分类
            >>> probs = results[0].probs  # 获取分类概率
            >>> top1_confidence = probs.top1conf  # 获取 top 1 类别的置信度
            >>> print(f"Top 1 class confidence: {top1_confidence.item():.4f}")
        """
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self) -> torch.Tensor | np.ndarray:
        """返回前 5 个分类预测的置信度分数。

        该属性检索与模型预测的前 5 个类别概率对应的置信度分数。
        它提供了一种快速访问最可能的类别预测及其相关置信度级别的方法。

        Returns:
            (torch.Tensor | np.ndarray): 包含前 5 个预测类别的置信度分数的张量或数组,按概率降序排列。

        Examples:
            >>> results = model("image.jpg")
            >>> probs = results[0].probs
            >>> top5_conf = probs.top5conf
            >>> print(top5_conf)  # 打印前 5 个类别的置信度分数
        """
        return self.data[self.top5]


class OBB(BaseTensor):
    """用于存储和操作有向边界框 (OBB) 的类。

    该类提供处理有向边界框的功能,包括不同格式之间的转换、归一化以及对边界框各种属性的访问。
    它支持跟踪和非跟踪场景。

    Attributes:
        data (torch.Tensor): 包含边界框坐标和关联数据的原始 OBB 张量。
        orig_shape (tuple): 原始图像大小,格式为 (高度, 宽度)。
        is_track (bool): 指示边界框数据中是否包含跟踪 ID。
        xywhr (torch.Tensor | np.ndarray): [x_center, y_center, width, height, rotation] 格式的边界框。
        conf (torch.Tensor | np.ndarray): 每个边界框的置信度分数。
        cls (torch.Tensor | np.ndarray): 每个边界框的类别标签。
        id (torch.Tensor | np.ndarray): 每个边界框的跟踪 ID(如果可用)。
        xyxyxyxy (torch.Tensor | np.ndarray): 8 点 [x1, y1, x2, y2, x3, y3, x4, y4] 格式的边界框。
        xyxyxyxyn (torch.Tensor | np.ndarray): 相对于 orig_shape 归一化的 8 点坐标。
        xyxy (torch.Tensor | np.ndarray): [x1, y1, x2, y2] 格式的轴对齐边界框。

    Methods:
        cpu: 返回所有张量在 CPU 内存上的 OBB 对象副本。
        numpy: 返回所有张量为 numpy 数组的 OBB 对象副本。
        cuda: 返回所有张量在 GPU 内存上的 OBB 对象副本。
        to: 返回张量在指定设备和数据类型上的 OBB 对象副本。

    Examples:
        >>> boxes = torch.tensor([[100, 50, 150, 100, 30, 0.9, 0]])  # xywhr, conf, cls
        >>> obb = OBB(boxes, orig_shape=(480, 640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
        >>> print(obb.cls)
    """

    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """使用有向边界框数据和原始图像形状初始化 OBB(有向边界框)实例。

        该类存储和操作用于目标检测任务的有向边界框 (OBB)。它提供访问和转换 OBB 数据的各种属性和方法。

        Args:
            boxes (torch.Tensor | np.ndarray): 包含检测框的张量或 numpy 数组,形状为 (num_boxes, 7) 或 (num_boxes, 8)。
                最后两列包含置信度和类别值。如果存在,倒数第三列包含跟踪 ID,第五列包含旋转角度。
            orig_shape (tuple[int, int]): 原始图像大小,格式为 (高度, 宽度)。

        Raises:
            AssertionError: 如果每个边界框的值数量不是 7 或 8。
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> torch.Tensor | np.ndarray:
        """返回 [x_center, y_center, width, height, rotation] 格式的边界框。

        Returns:
            (torch.Tensor | np.ndarray): 包含有向边界框的张量或 numpy 数组,
                格式为 [x_center, y_center, width, height, rotation]。形状为 (N, 5),其中 N 是边界框数量。

        Examples:
            >>> results = model("image.jpg")
            >>> obb = results[0].obb
            >>> xywhr = obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3, 5])
        """
        return self.data[:, :5]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """返回有向边界框 (OBB) 的置信度分数。

        该属性检索与每个 OBB 检测相关的置信度值。置信度分数表示模型对检测的确定性。

        Returns:
            (torch.Tensor | np.ndarray): 形状为 (N,) 的张量或 numpy 数组,包含 N 个检测的置信度分数,
                每个分数在 [0, 1] 范围内。

        Examples:
            >>> results = model("image.jpg")
            >>> obb_result = results[0].obb
            >>> confidence_scores = obb_result.conf
            >>> print(confidence_scores)
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """返回有向边界框的类别值。

        Returns:
            (torch.Tensor | np.ndarray): 包含每个有向边界框的类别值的张量或 numpy 数组。
                形状为 (N,),其中 N 是边界框数量。

        Examples:
            >>> results = model("image.jpg")
            >>> result = results[0]
            >>> obb = result.obb
            >>> class_values = obb.cls
            >>> print(class_values)
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray | None:
        """返回有向边界框的跟踪 ID(如果可用)。

        Returns:
            (torch.Tensor | np.ndarray | None): 包含每个有向边界框的跟踪 ID 的张量或 numpy 数组。
                如果跟踪 ID 不可用,则返回 None。

        Examples:
            >>> results = model("image.jpg", tracker=True)  # 使用跟踪运行推理
            >>> for result in results:
            ...     if result.obb is not None:
            ...         track_ids = result.obb.id
            ...         if track_ids is not None:
            ...             print(f"Tracking IDs: {track_ids}")
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self) -> torch.Tensor | np.ndarray:
        """将 OBB 格式转换为旋转边界框的 8 点 (xyxyxyxy) 坐标格式。

        Returns:
            (torch.Tensor | np.ndarray): xyxyxyxy 格式的旋转边界框,形状为 (N, 4, 2),其中 N 是边界框数量。
                每个边界框由 4 个点 (x, y) 表示,从左上角开始顺时针移动。

        Examples:
            >>> obb = OBB(torch.tensor([[100, 100, 50, 30, 0.5, 0.9, 0]]), orig_shape=(640, 640))
            >>> xyxyxyxy = obb.xyxyxyxy
            >>> print(xyxyxyxy.shape)
            torch.Size([1, 4, 2])
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self) -> torch.Tensor | np.ndarray:
        """将旋转边界框转换为归一化的 xyxyxyxy 格式。

        Returns:
            (torch.Tensor | np.ndarray): xyxyxyxy 格式的归一化旋转边界框,形状为 (N, 4, 2),
                其中 N 是边界框数量。每个边界框由 4 个点 (x, y) 表示,相对于原始图像尺寸进行归一化。

        Examples:
            >>> obb = OBB(torch.rand(10, 7), orig_shape=(640, 480))  # 10 个随机 OBB
            >>> normalized_boxes = obb.xyxyxyxyn
            >>> print(normalized_boxes.shape)
            torch.Size([10, 4, 2])
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """将有向边界框 (OBB) 转换为 xyxy 格式的轴对齐边界框。

        该属性计算每个有向边界框的最小外接矩形,并以 xyxy 格式 (x1, y1, x2, y2) 返回。
        这对于需要轴对齐边界框的操作很有用,例如与非旋转边界框的 IoU 计算。

        Returns:
            (torch.Tensor | np.ndarray): xyxy 格式的轴对齐边界框,形状为 (N, 4),其中 N 是边界框数量。
                每行包含 [x1, y1, x2, y2] 坐标。

        Examples:
            >>> import torch
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n-obb.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     obb = result.obb
            ...     if obb is not None:
            ...         xyxy_boxes = obb.xyxy
            ...         print(xyxy_boxes.shape)  # (N, 4)

        Notes:
            - 该方法通过最小外接矩形近似 OBB。
            - 返回的格式与标准目标检测指标和可视化工具兼容。
            - 该属性使用缓存以提高重复访问的性能。
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )
