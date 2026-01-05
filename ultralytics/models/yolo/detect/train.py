"""
YOLO 目标检测训练模块

该模块实现了 YOLO 目标检测模型的训练功能,包括:
    - 数据集构建和数据加载
    - 批量数据预处理 (图像缩放、归一化、多尺度训练)
    - 模型属性设置 (类别数、类别名称等)
    - 训练过程监控和可视化
    - 自动批量大小计算

主要类:
    - DetectionTrainer: 目标检测训练器,继承自 BaseTrainer

训练流程:
    1. 构建训练数据集 (build_dataset)
    2. 创建数据加载器 (get_dataloader)
    3. 预处理批量数据 (preprocess_batch)
    4. 前向传播计算损失 (box_loss, cls_loss, dfl_loss)
    5. 反向传播更新权重
    6. 验证和可视化

典型应用:
    - 训练 YOLO 检测模型
    - 微调预训练模型
    - 多尺度训练提升精度
    - 分布式训练加速
"""

from __future__ import annotations  # 启用延迟类型注解评估

import math  # 数学函数
import random  # 随机数生成
from copy import copy  # 浅拷贝
from typing import Any  # 类型提示

import numpy as np  # 数值计算
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络模块

# 导入数据处理相关模块
from ultralytics.data import build_dataloader, build_yolo_dataset  # 数据加载器和数据集构建
from ultralytics.engine.trainer import BaseTrainer  # 训练器基类
from ultralytics.models import yolo  # YOLO 模型模块
from ultralytics.nn.tasks import DetectionModel  # 检测模型架构
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK  # 配置、日志、进程rank
from ultralytics.utils.patches import override_configs  # 配置覆盖工具
from ultralytics.utils.plotting import plot_images, plot_labels  # 可视化工具
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model  # PyTorch 工具


class DetectionTrainer(BaseTrainer):
    """扩展 BaseTrainer 类的目标检测训练器类

    该训练器专门处理目标检测任务,处理 YOLO 模型训练的特定需求,包括数据集构建、
    数据加载、预处理和模型配置。

    属性:
        model (DetectionModel): 正在训练的 YOLO 检测模型
        data (dict): 包含数据集信息的字典,包括类别名称和类别数量
        loss_names (tuple): 训练中使用的损失组件名称 (box_loss, cls_loss, dfl_loss)

    方法:
        build_dataset: 构建用于训练或验证的 YOLO 数据集
        get_dataloader: 构造并返回指定模式的数据加载器
        preprocess_batch: 通过缩放和转换为浮点数预处理批量图像
        set_model_attributes: 根据数据集信息设置模型属性
        get_model: 返回 YOLO 检测模型
        get_validator: 返回用于模型评估的验证器
        label_loss_items: 返回带有标记的训练损失项的损失字典
        progress_string: 返回格式化的训练进度字符串
        plot_training_samples: 绘制训练样本及其标注
        plot_training_labels: 创建 YOLO 模型的标记训练图
        auto_batch: 根据模型内存需求计算最佳批量大小

    示例:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化用于训练 YOLO 目标检测模型的 DetectionTrainer 对象

        初始化目标检测训练器

        参数:
            cfg (dict, optional): 包含训练参数的默认配置字典
                包含学习率、批量大小、epochs 等训练参数
            overrides (dict, optional): 用于覆盖默认配置的参数字典
            _callbacks (list, optional): 训练过程中执行的回调函数列表
        """
        # 调用父类初始化方法
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集

        构建 YOLO 数据集

        根据模式 (训练/验证) 构建数据集,并应用相应的数据增强策略:
            - train 模式: 使用 Mosaic、MixUp、随机翻转等增强
            - val 模式: 使用矩形推理 (rect=True) 提高验证速度

        参数:
            img_path (str): 包含图像的文件夹路径
            mode (str): 模式,'train' (训练) 或 'val' (验证),用户可为每种模式自定义不同的增强
            batch (int, optional): 批量大小,用于矩形推理模式

        返回:
            (Dataset): 配置好的 YOLO 数据集对象
        """
        # 获取模型的最大步长,用于确保图像尺寸是步长的倍数
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        # 构建并返回 YOLO 数据集 (验证模式使用矩形推理)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """构造并返回指定模式的数据加载器

        参数:
            dataset_path (str): 数据集路径
            batch_size (int): 每批图像数量
            rank (int): 分布式训练的进程 rank
            mode (str): 'train' 表示训练数据加载器,'val' 表示验证数据加载器

        返回:
            (DataLoader): PyTorch 数据加载器对象
        """
        assert mode in {"train", "val"}, f"模式必须是 'train' 或 'val',而不是 {mode}。"
        with torch_distributed_zero_first(rank):  # 如果使用 DDP,仅初始化数据集 *.cache 一次
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' 与 DataLoader shuffle 不兼容,设置 shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """通过缩放和转换为浮点数预处理批量图像

        预处理批量图像

        对批量数据进行预处理,包括:
            1. 将数据移动到目标设备 (GPU/CPU)
            2. 图像归一化到 [0, 1] 范围 (除以 255)
            3. 多尺度训练 (如果启用): 随机调整图像尺寸

        参数:
            batch (dict): 包含批量数据的字典,包括 'img' (图像张量)、'bboxes' (边界框) 等

        返回:
            (dict): 预处理后的批量数据字典,包含归一化图像
        """
        # 将所有张量数据移动到目标设备 (GPU/CPU)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # 使用 non_blocking=True 加速 GPU 传输
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        # 图像归一化: 将像素值从 [0, 255] 缩放到 [0, 1]
        batch["img"] = batch["img"].float() / 255
        # 多尺度训练: 随机调整图像尺寸以提高模型鲁棒性
        if self.args.multi_scale:
            imgs = batch["img"]
            # 随机选择尺寸 (0.5x ~ 1.5x 的 imgsz),并确保是 stride 的倍数
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # 计算缩放因子
            if sf != 1:
                # 计算新的图像尺寸 (确保是 stride 的倍数)
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                # 使用双线性插值调整图像大小
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information.

        设置模型属性

        根据数据集信息设置模型的关键属性:
            - nc: 类别数量
            - names: 类别名称列表
            - args: 训练超参数
        """
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # 设置模型的类别数量
        self.model.names = self.data["names"]  # 设置模型的类别名称
        self.model.args = self.args  # 设置模型的训练参数
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        获取 YOLO 检测模型

        Args:
            cfg (str, optional): Path to model configuration file.
                模型配置文件路径 (YAML 文件)
            weights (str, optional): Path to model weights.
                模型权重文件路径 (PT 文件)
            verbose (bool): Whether to display model information.
                是否显示模型信息

        Returns:
            (DetectionModel): YOLO detection model.
                YOLO 检测模型实例
        """
        # 创建检测模型 (指定类别数和输入通道数)
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        # 如果提供了权重文件,则加载权重
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation.

        获取检测验证器

        创建并返回用于模型验证的 DetectionValidator 实例。
        同时定义了三个损失名称: box_loss (边界框损失)、cls_loss (分类损失)、dfl_loss (分布焦点损失)。

        Returns:
            (DetectionValidator): 检测验证器实例
        """
        # 定义损失名称: 边界框损失、分类损失、分布焦点损失
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        # 创建并返回检测验证器
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """返回带有标记的训练损失项的损失字典

        参数:
            loss_items (list[float], optional): 损失值列表
            prefix (str): 返回字典中键的前缀

        返回:
            (dict | list): 如果提供了 loss_items,则返回标记的损失项字典,否则返回键列表
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为 5 位小数的浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回格式化的训练进度字符串,包含 epoch、GPU 内存、损失、实例数和尺寸"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制训练样本及其标注

        参数:
            batch (dict[str, Any]): 包含批量数据的字典
            ni (int): 迭代次数
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建 YOLO 模型的标记训练图"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """通过计算模型的内存占用获取最佳批量大小

        返回:
            (int): 最佳批量大小
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 用于 mosaic 增强
        del train_dataset  # 释放内存
        return super().auto_batch(max_num_obj)
