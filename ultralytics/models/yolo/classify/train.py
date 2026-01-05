"""
YOLO 图像分类训练模块

该模块实现了 YOLO 图像分类模型的训练功能,包括:
    - 分类数据集构建和加载
    - 图像分类模型训练
    - Top-1/Top-5 准确率验证
    - 学习率调度和优化

主要类:
    - ClassificationTrainer: 图像分类训练器,继承自 BaseTrainer

训练流程:
    1. 加载分类数据集 (ImageNet, 自定义数据集)
    2. 构建分类模型 (YOLO backbone + 分类头)
    3. 前向传播计算交叉熵损失
    4. 反向传播更新权重
    5. 验证 Top-1/Top-5 准确率

损失函数:
    - CrossEntropyLoss: 交叉熵损失

典型应用:
    - ImageNet 预训练
    - 自定义分类任务
    - 迁移学习
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import copy  # 浅拷贝
from typing import Any  # 类型提示

import torch  # PyTorch 深度学习框架

from ultralytics.data import ClassificationDataset, build_dataloader  # 分类数据集和数据加载器
from ultralytics.engine.trainer import BaseTrainer  # 训练器基类
from ultralytics.models import yolo  # YOLO 模型模块
from ultralytics.nn.tasks import ClassificationModel  # 分类模型架构
from ultralytics.utils import DEFAULT_CFG, RANK  # 配置和进程rank
from ultralytics.utils.plotting import plot_images  # 可视化工具
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first  # PyTorch 工具


class ClassificationTrainer(BaseTrainer):
    """用于训练图像分类模型的训练器类，扩展 BaseTrainer。

    该训练器处理图像分类任务的训练过程，支持 YOLO 分类模型和 torchvision 模型，
    具有全面的数据集处理和验证功能。

    属性:
        model (ClassificationModel): 要训练的分类模型。
        data (dict[str, Any]): 包含数据集信息的字典，包括类别名称和类别数量。
        loss_names (list[str]): 训练期间使用的损失函数名称。
        validator (ClassificationValidator): 用于模型评估的验证器实例。

    方法:
        set_model_attributes: 从加载的数据集设置模型的类别名称。
        get_model: 返回配置用于训练的修改后 PyTorch 模型。
        setup_model: 加载、创建或下载用于分类的模型。
        build_dataset: 创建 ClassificationDataset 实例。
        get_dataloader: 返回带有图像预处理变换的 PyTorch DataLoader。
        preprocess_batch: 预处理一批图像和类别。
        progress_string: 返回显示训练进度的格式化字符串。
        get_validator: 返回 ClassificationValidator 实例。
        label_loss_items: 返回带有标记的训练损失项的损失字典。
        final_eval: 评估训练后的模型并保存验证结果。
        plot_training_samples: 绘制训练样本及其标注。

    示例:
        初始化并训练分类模型
        >>> from ultralytics.models.yolo.classify import ClassificationTrainer
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10", epochs=3)
        >>> trainer = ClassificationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化 ClassificationTrainer 对象。

        参数:
            cfg (dict[str, Any], optional): 包含训练参数的默认配置字典。
            overrides (dict[str, Any], optional): 默认配置的参数覆盖字典。
            _callbacks (list[Any], optional): 训练期间要执行的回调函数列表。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """从加载的数据集设置 YOLO 模型的类别名称。"""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回配置用于训练 YOLO 分类的修改后 PyTorch 模型。

        参数:
            cfg (Any, optional): 模型配置。
            weights (Any, optional): 预训练模型权重。
            verbose (bool, optional): 是否显示模型信息。

        返回:
            (ClassificationModel): 配置用于分类的 PyTorch 模型。
        """
        model = ClassificationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # 设置 dropout
        for p in model.parameters():
            p.requires_grad = True  # 用于训练
        return model

    def setup_model(self):
        """加载、创建或下载用于分类任务的模型。

        返回:
            (Any): 如果适用则返回模型检查点，否则返回 None。
        """
        import torchvision  # 限制作用域以加快 'import ultralytics'

        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """根据给定的图像路径和模式创建 ClassificationDataset 实例。

        参数:
            img_path (str): 数据集图像的路径。
            mode (str, optional): 数据集模式（'train'、'val' 或 'test'）。
            batch (Any, optional): 批次信息（在此实现中未使用）。

        返回:
            (ClassificationDataset): 指定模式的数据集。
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """返回带有图像预处理变换的 PyTorch DataLoader。

        参数:
            dataset_path (str): 数据集的路径。
            batch_size (int, optional): 每批图像数量。
            rank (int, optional): 分布式训练的进程rank。
            mode (str, optional): 'train'、'val' 或 'test' 模式。

        返回:
            (torch.utils.data.DataLoader): 指定数据集和模式的 DataLoader。
        """
        with torch_distributed_zero_first(rank):  # 如果使用 DDP，仅初始化数据集 *.cache 一次
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)
        # 附加推理变换
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """预处理一批图像和类别。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def progress_string(self) -> str:
        """返回显示训练进度的格式化字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """返回用于验证的 ClassificationValidator 实例。"""
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: torch.Tensor | None = None, prefix: str = "train"):
        """返回带有标记的训练损失项张量的损失字典。

        参数:
            loss_items (torch.Tensor, optional): 损失张量项。
            prefix (str, optional): 添加到损失名称前面的前缀。

        返回:
            keys (list[str]): 如果 loss_items 为 None，则返回损失键列表。
            loss_dict (dict[str, float]): 如果提供 loss_items，则返回损失项字典。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_training_samples(self, batch: dict[str, torch.Tensor], ni: int):
        """绘制训练样本及其标注。

        参数:
            batch (dict[str, torch.Tensor]): 包含图像和类别标签的批次。
            ni (int): 迭代次数。
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # 添加批次索引用于绘图
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
