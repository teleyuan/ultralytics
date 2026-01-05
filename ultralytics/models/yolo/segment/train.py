"""
YOLO 实例分割训练模块

该模块实现了 YOLO 实例分割模型的训练功能,继承自检测训练器并扩展了分割功能:
    - 边界框和分割掩码联合训练
    - 掩码损失计算 (BCE + Dice Loss)
    - 分割结果可视化

主要类:
    - SegmentationTrainer: 实例分割训练器,继承自 DetectionTrainer

训练流程:
    1. 加载分割数据集 (包含边界框和分割掩码)
    2. 构建分割模型 (检测头 + 分割头)
    3. 计算多任务损失 (box_loss + cls_loss + dfl_loss + mask_loss)
    4. 反向传播更新权重
    5. 验证 mAP (box) 和 mAP (mask)

损失函数:
    - box_loss: 边界框回归损失
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失
    - mask_loss: 掩码损失 (BCE + Dice)

典型应用:
    - COCO 实例分割
    - 自定义分割任务
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import copy  # 浅拷贝
from pathlib import Path  # 路径操作

from ultralytics.models import yolo  # YOLO 模型模块
from ultralytics.nn.tasks import SegmentationModel  # 分割模型架构
from ultralytics.utils import DEFAULT_CFG, RANK  # 配置和进程rank


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """扩展 DetectionTrainer 类的分割模型训练器类

    该训练器专门处理分割任务,通过分割特定功能(包括模型初始化、验证和可视化)扩展检测训练器。

    属性:
        loss_names (tuple[str]): 训练期间使用的损失组件名称

    示例:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """初始化 SegmentationTrainer 对象

        参数:
            cfg (dict): 包含默认训练设置的配置字典
            overrides (dict, optional): 默认配置的参数覆盖字典
            _callbacks (list, optional): 训练期间执行的回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """初始化并返回具有指定配置和权重的 SegmentationModel

        参数:
            cfg (dict | str, optional): 模型配置。可以是字典、YAML 文件路径或 None
            weights (str | Path, optional): 预训练权重文件的路径
            verbose (bool): 是否在初始化期间显示模型信息

        返回:
            (SegmentationModel): 初始化后的分割模型,如果指定则加载权重

        示例:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo11n-seg.pt", verbose=False)
        """
        model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 SegmentationValidator 实例"""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
