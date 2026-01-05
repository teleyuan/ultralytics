"""
YOLO 有向边界框检测训练模块 (OBB - Oriented Bounding Box)

该模块实现了 YOLO 有向边界框检测模型的训练功能,继承自检测训练器并扩展了旋转框功能:
    - 支持任意角度的边界框预测
    - Probiou Loss 用于旋转框回归
    - 适用于遥感图像和文本检测

主要类:
    - OBBTrainer: 有向边界框训练器,继承自 DetectionTrainer

训练流程:
    1. 加载 OBB 数据集 (包含旋转角度的边界框)
    2. 构建 OBB 模型 (检测头 + 角度预测)
    3. 计算多任务损失 (box_loss + cls_loss + dfl_loss)
    4. 反向传播更新权重
    5. 验证 Rotated mAP

损失函数:
    - box_loss: 有向边界框回归损失 (Probiou Loss)
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失

典型应用:
    - DOTA 遥感图像检测
    - 文本检测
    - 密集场景检测
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import copy  # 浅拷贝
from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

from ultralytics.models import yolo  # YOLO 模型模块
from ultralytics.nn.tasks import OBBModel  # OBB 模型架构
from ultralytics.utils import DEFAULT_CFG, RANK  # 配置和进程rank


class OBBTrainer(yolo.detect.DetectionTrainer):
    """扩展 DetectionTrainer 类的有向边界框 (OBB) 训练器类

    该训练器专门用于训练检测有向边界框的 YOLO 模型,适用于检测任意角度的目标,
    而不仅仅是轴对齐的矩形框。

    属性:
        loss_names (tuple): 训练中使用的损失组件名称,包括 box_loss、cls_loss 和 dfl_loss

    方法:
        get_model: 返回使用指定配置和权重初始化的 OBBModel
        get_validator: 返回用于 YOLO 模型验证的 OBBValidator 实例

    示例:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        """初始化用于训练有向边界框 (OBB) 模型的 OBBTrainer 对象

        参数:
            cfg (dict, optional): 训练器的配置字典,包含训练参数和模型配置
            overrides (dict, optional): 配置的参数覆盖字典,此处的任何值将优先于 cfg 中的值
            _callbacks (list[Any], optional): 训练期间调用的回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """返回使用指定配置和权重初始化的 OBBModel

        参数:
            cfg (str | dict, optional): 模型配置,可以是 YAML 配置文件的路径、包含配置参数的字典,
                或 None 以使用默认配置
            weights (str | Path, optional): 预训练权重文件的路径,如果为 None,则使用随机初始化
            verbose (bool): 是否在初始化期间显示模型信息

        返回:
            (OBBModel): 使用指定配置和权重初始化的 OBBModel

        示例:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-obb.yaml", weights="yolo11n-obb.pt")
        """
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 OBBValidator 实例"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
