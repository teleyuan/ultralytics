"""
YOLOE 增强型检测训练模块

该模块实现了 YOLOE 增强型检测模型的多种训练模式:
    - 标准训练: 使用文本提示进行训练
    - 视觉提示训练: 使用参考图像中的目标作为检测模板
    - 位置编码训练: 支持位置编码和无位置编码训练
    - 从头训练: 不使用预训练权重

主要类:
    - YOLOETrainer: 标准 YOLOE 检测训练器
    - YOLOETrainerFromScratch: 从头开始训练 YOLOE
    - YOLOEVPTrainer: 视觉提示训练器
    - YOLOEPETrainer: 位置编码训练器
    - YOLOEPEFreeTrainer: 无位置编码训练器

训练流程:
    1. 加载检测数据集 (支持多模态)
    2. 构建 YOLOEModel (支持文本/视觉提示)
    3. 计算多任务损失
    4. 更新模型权重
    5. 验证检测性能

关键特性:
    - 多模态支持: 文本提示和视觉提示
    - 灵活训练: 支持多种训练模式
    - 位置编码: 可选的位置编码支持

典型应用:
    - 少样本检测
    - 视觉提示检测
    - 灵活类别检测
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import copy, deepcopy  # 拷贝工具
from pathlib import Path  # 路径操作

import torch  # PyTorch 深度学习框架

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset  # 数据集构建
from ultralytics.data.augment import LoadVisualPrompt  # 视觉提示加载
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator  # 检测训练器和验证器
from ultralytics.nn.tasks import YOLOEModel  # YOLOE 模型架构
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK  # 配置、日志、进程rank
from ultralytics.utils.torch_utils import unwrap_model  # 模型解包工具

from ..world.train_world import WorldTrainerFromScratch  # World 从头训练器
from .val import YOLOEDetectValidator  # YOLOE 检测验证器


class YOLOETrainer(DetectionTrainer):
    """YOLOE 目标检测模型的训练器类

    该类扩展了 DetectionTrainer,为 YOLOE 模型提供专门的训练功能,包括自定义模型初始化、验证和支持多模态的数据集构建。

    属性:
        loss_names (tuple): 训练期间使用的损失组件名称

    方法:
        get_model: 初始化并返回使用指定配置的 YOLOEModel
        get_validator: 返回用于模型验证的 YOLOEDetectValidator
        build_dataset: 构建支持多模态训练的 YOLO 数据集
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """使用指定配置初始化 YOLOE 训练器

        参数:
            cfg (dict): 包含 DEFAULT_CFG 默认训练设置的配置字典
            overrides (dict, optional): 默认配置的参数覆盖字典
            _callbacks (list, optional): 训练期间应用的回调函数列表
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"使用 'model={overrides['model']}' 训练需要 'compile=False'"
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回使用指定配置和权重初始化的 YOLOEModel

        参数:
            cfg (dict | str, optional): 模型配置。可以是包含 'yaml_file' 键的字典、YAML 文件的直接路径,
                或 None 以使用默认配置。
            weights (str | Path, optional): 要加载到模型中的预训练权重文件路径
            verbose (bool): 是否在初始化期间显示模型信息

        返回:
            (YOLOEModel): 初始化的 YOLOE 模型

        注意:
            - 按照官方配置,类别数 (nc) 硬编码为最大 80
            - 这里的 nc 参数表示一张图像中不同文本样本的最大数量,而不是实际的类别数
        """
        # 注意: 这里的 `nc` 是一张图像中不同文本样本的最大数量,而不是实际的 `nc`
        # 注意: 按照官方配置,nc 目前硬编码为 80
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLOE 模型验证的 YOLOEDetectValidator"""
        self.loss_names = "box", "cls", "dfl"
        return YOLOEDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建 YOLO 数据集

        参数:
            img_path (str): 包含图像的文件夹路径
            mode (str): 'train' 模式或 'val' 模式,用户可以为每个模式自定义不同的数据增强
            batch (int, optional): 批次大小,用于矩形训练

        返回:
            (Dataset): 配置用于训练或验证的 YOLO 数据集
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )


class YOLOEPETrainer(DetectionTrainer):
    """使用线性探测方法微调 YOLOE 模型

    该训练器冻结大部分模型层,仅训练特定的投影层,以便在保留预训练特征的同时对新数据集进行高效微调。

    方法:
        get_model: 初始化 YOLOEModel,除投影层外冻结其他层
    """

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回使用指定配置和权重初始化的 YOLOEModel

        参数:
            cfg (dict | str, optional): 模型配置
            weights (str, optional): 预训练权重的路径
            verbose (bool): 是否显示模型信息

        返回:
            (YOLOEModel): 初始化的模型,除特定投影层外其他层均冻结
        """
        # 注意: 这里的 `nc` 是一张图像中不同文本样本的最大数量,而不是实际的 `nc`
        # 注意: 按照官方配置,nc 目前硬编码为 80
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "线性探测必须提供预训练权重。"
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # 注意: `get_text_pe` 与文本模型和 YOLOEDetect.reprta 相关,
        # 只要加载正确的预训练权重就能得到正确结果
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)  # 将文本嵌入融合到分类头
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):
    """支持文本嵌入的从头开始训练 YOLOE 模型

    该训练器结合了 YOLOE 训练功能和 world 训练特性,支持使用文本嵌入和 grounding 数据集从头开始训练。

    方法:
        build_dataset: 构建支持 grounding 的训练数据集
        generate_text_embeddings: 生成并缓存用于训练的文本嵌入
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集

        该方法根据模式和输入路径构建适当的数据集,处理标准 YOLO 数据集和不同格式的 grounding 数据集。

        参数:
            img_path (list[str] | str): 包含图像的文件夹路径或路径列表
            mode (str): 'train' 模式或 'val' 模式,允许为每个模式自定义数据增强
            batch (int, optional): 批次大小,用于矩形训练/验证

        返回:
            (YOLOConcatDataset | Dataset): 构建的用于训练或验证的数据集
        """
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        """为文本样本列表生成文本嵌入

        参数:
            texts (list[str]): 要编码的文本样本列表
            batch (int): 处理的批次大小
            cache_dir (Path): 保存/加载缓存嵌入的目录

        返回:
            (dict): 将文本样本映射到其嵌入的字典
        """
        model = "mobileclip:blt"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"从 '{cache_path}' 读取已存在的缓存")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"将文本嵌入缓存到 '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    """训练无提示 YOLOE 模型

    该训练器结合了线性探测功能和从头训练功能,用于推理期间不需要文本提示的无提示 YOLOE 模型。

    方法:
        get_validator: 返回用于验证的标准 DetectionValidator
        preprocess_batch: 预处理不带文本特征的批次
        set_text_embeddings: 为数据集设置文本嵌入 (无提示模式下为空操作)
    """

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator"""
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """为 YOLOE 训练预处理一批图像,根据需要调整格式和尺寸"""
        return DetectionTrainer.preprocess_batch(self, batch)

    def set_text_embeddings(self, datasets, batch: int):
        """通过缓存类别名称为数据集设置文本嵌入以加速训练

        该方法从所有数据集收集唯一的类别名称,为它们生成文本嵌入,并缓存这些嵌入以提高训练效率。
        嵌入存储在第一个数据集图像路径的父目录中的文件中。

        参数:
            datasets (list[Dataset]): 包含要处理的类别名称的数据集列表
            batch (int): 处理文本嵌入的批次大小

        注意:
            该方法创建一个将文本样本映射到其嵌入的字典,并将其存储在 'cache_path' 指定的路径。
            如果缓存文件已存在,将加载它而不是重新生成嵌入。
        """
        pass


class YOLOEVPTrainer(YOLOETrainerFromScratch):
    """使用视觉提示训练 YOLOE 模型

    该训练器扩展了 YOLOETrainerFromScratch 以支持基于视觉提示的训练,在图像旁提供视觉线索来指导检测过程。

    方法:
        build_dataset: 构建包含视觉提示加载变换的数据集
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """构建包含视觉提示的用于训练或验证的 YOLO 数据集

        参数:
            img_path (list[str] | str): 包含图像的文件夹路径或路径列表
            mode (str): 'train' 模式或 'val' 模式,允许为每个模式自定义数据增强
            batch (int, optional): 批次大小,用于矩形训练/验证

        返回:
            (Dataset): 配置用于训练或验证的 YOLO 数据集,训练模式包含视觉提示
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def _close_dataloader_mosaic(self):
        """关闭 mosaic 数据增强并将视觉提示加载添加到训练数据集"""
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, YOLOConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())
