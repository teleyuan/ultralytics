"""
YOLO-World 开放词汇目标检测训练模块

该模块实现了 YOLO-World 开放词汇检测模型的训练功能,结合视觉和文本特征:
    - 文本嵌入生成和缓存
    - 视觉-语言联合训练
    - 支持零样本和少样本检测
    - 自定义词汇表管理

主要类:
    - WorldTrainer: 开放词汇检测训练器,继承自 DetectionTrainer

训练流程:
    1. 加载检测数据集和文本嵌入
    2. 构建 WorldModel (视觉编码器 + 文本编码器)
    3. 计算多任务损失 (box_loss + cls_loss + dfl_loss)
    4. 文本-视觉特征对齐
    5. 验证开放词汇检测性能

关键特性:
    - 文本提示: 支持自定义类别文本描述
    - CLIP 集成: 使用 CLIP 模型生成文本嵌入
    - 开放词汇: 无需预定义类别,支持任意文本

典型应用:
    - 零样本目标检测
    - 开放域检测
    - 灵活类别检测
"""

from __future__ import annotations  # 启用延迟类型注解评估

import itertools  # 迭代工具
from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

import torch  # PyTorch 深度学习框架

from ultralytics.data import build_yolo_dataset  # YOLO 数据集构建
from ultralytics.models.yolo.detect import DetectionTrainer  # 检测训练器基类
from ultralytics.nn.tasks import WorldModel  # World 模型架构
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK  # 配置、日志、进程rank
from ultralytics.utils.torch_utils import unwrap_model  # 模型解包工具


def on_pretrain_routine_end(trainer) -> None:
    """在预训练例程结束时设置模型类别和文本编码器

    在预训练结束后设置模型类别名称和文本编码器,为评估做准备。

    参数:
        trainer: 训练器实例
    """
    if RANK in {-1, 0}:  # 只在主进程或单进程中执行
        # 从数据集中提取类别名称 (去除父类别信息)
        names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        # 为 EMA 模型设置类别名称 (不缓存 CLIP 模型)
        unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


class WorldTrainer(DetectionTrainer):
    """在封闭集数据集上微调 YOLO World 模型的训练器类

    该训练器扩展了 DetectionTrainer 以支持训练 YOLO World 模型,该模型结合视觉和文本特征以改进目标检测和理解。
    它处理文本嵌入生成和缓存,以加速多模态数据的训练。

    属性:
        text_embeddings (dict[str, torch.Tensor] | None): 缓存的类别名称文本嵌入,用于加速训练
        model (WorldModel): 正在训练的 YOLO World 模型
        data (dict[str, Any]): 包含类别信息的数据集配置
        args (Any): 训练参数和配置

    方法:
        get_model: 返回使用指定配置和权重初始化的 WorldModel
        build_dataset: 构建用于训练或验证的 YOLO 数据集
        set_text_embeddings: 为数据集设置文本嵌入以加速训练
        generate_text_embeddings: 为文本样本列表生成文本嵌入
        preprocess_batch: 为 YOLOWorld 训练预处理图像和文本批次

    示例:
        初始化并训练 YOLO World 模型
        >>> from ultralytics.models.yolo.world import WorldTrainer
        >>> args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        >>> trainer = WorldTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """使用给定参数初始化 WorldTrainer 对象

        参数:
            cfg (dict[str, Any]): 训练器的配置
            overrides (dict[str, Any], optional): 配置覆盖
            _callbacks (list[Any], optional): 回调函数列表
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"使用 'model={overrides['model']}' 训练需要 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WorldModel:
        """返回使用指定配置和权重初始化的 WorldModel

        参数:
            cfg (dict[str, Any] | str, optional): 模型配置
            weights (str, optional): 预训练权重的路径
            verbose (bool): 是否显示模型信息

        返回:
            (WorldModel): 初始化的 WorldModel
        """
        # 注意: 这里的 `nc` 是一张图像中不同文本样本的最大数量,而不是实际的 `nc`
        # 注意: 按照官方配置,nc 目前硬编码为 80
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建用于训练或验证的 YOLO 数据集

        参数:
            img_path (str): 包含图像的文件夹路径
            mode (str): `train` 模式或 `val` 模式,用户可以为每个模式自定义不同的数据增强
            batch (int, optional): 批次大小,用于 `rect`

        返回:
            (Any): 配置用于训练或验证的 YOLO 数据集
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)  # 缓存文本嵌入以加速训练
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        """通过缓存类别名称为数据集设置文本嵌入以加速训练

        该方法从所有数据集收集唯一的类别名称,然后为这些类别生成并缓存文本嵌入以提高训练效率。

        参数:
            datasets (list[Any]): 从中提取类别名称的数据集列表
            batch (int | None): 用于处理的批次大小

        注意:
            该方法从具有 'category_names' 属性的数据集收集类别名称,然后使用第一个数据集的图像路径
            来确定缓存生成的文本嵌入的位置。
        """
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        """为文本样本列表生成文本嵌入

        参数:
            texts (list[str]): 要编码的文本样本列表
            batch (int): 处理的批次大小
            cache_dir (Path): 保存/加载缓存嵌入的目录

        返回:
            (dict[str, torch.Tensor]): 将文本样本映射到其嵌入的字典
        """
        model = "clip:ViT-B/32"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"从 '{cache_path}' 读取已存在的缓存")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"将文本嵌入缓存到 '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """为 YOLOWorld 训练预处理图像和文本批次"""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        # 添加文本特征
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
