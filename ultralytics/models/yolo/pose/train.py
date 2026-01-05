"""
YOLO 姿态估计训练模块

该模块实现了 YOLO 姿态估计模型的训练功能,继承自检测训练器并扩展了姿态估计功能:
    - 边界框和关键点联合训练
    - 关键点可见性预测
    - OKS (Object Keypoint Similarity) 损失计算

主要类:
    - PoseTrainer: 姿态估计训练器,继承自 DetectionTrainer

训练流程:
    1. 加载姿态估计数据集 (包含边界框、关键点和可见性)
    2. 构建姿态估计模型 (检测头 + 关键点头)
    3. 计算多任务损失 (box_loss + cls_loss + dfl_loss + kpt_loss)
    4. 反向传播更新权重
    5. 验证 mAP (box) 和 mAP (pose)

损失函数:
    - box_loss: 边界框回归损失
    - cls_loss: 分类损失
    - dfl_loss: 分布焦点损失
    - kpt_loss: 关键点损失 (OKS Loss)

典型应用:
    - COCO 人体姿态估计
    - 多人姿态估计
    - 关键点检测
"""

from __future__ import annotations  # 启用延迟类型注解评估

from copy import copy  # 浅拷贝
from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

from ultralytics.models import yolo  # YOLO 模型模块
from ultralytics.nn.tasks import PoseModel  # 姿态估计模型架构
from ultralytics.utils import DEFAULT_CFG  # 配置


class PoseTrainer(yolo.detect.DetectionTrainer):
    """扩展 DetectionTrainer 类的 YOLO 姿态估计训练器类

    该训练器专门处理姿态估计任务,管理模型训练、验证以及边界框和姿态关键点的可视化。

    属性:
        args (dict): 训练的配置参数
        model (PoseModel): 正在训练的姿态估计模型
        data (dict): 数据集配置,包含关键点形状信息
        loss_names (tuple): 训练中使用的损失组件名称

    方法:
        get_model: 获取具有指定配置的姿态估计模型
        set_model_attributes: 在模型上设置关键点形状属性
        get_validator: 创建用于模型评估的验证器实例
        plot_training_samples: 可视化带有关键点的训练样本
        get_dataset: 获取数据集并确保其包含必需的 kpt_shape 键

    示例:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """初始化用于训练 YOLO 姿态估计模型的 PoseTrainer 对象

        参数:
            cfg (dict, optional): 包含训练参数的默认配置字典
            overrides (dict, optional): 默认配置的参数覆盖字典
            _callbacks (list, optional): 训练期间执行的回调函数列表

        注意:
            该训练器将自动将任务设置为 'pose',无论 overrides 中提供了什么。
            使用 Apple MPS 设备时会发出警告,因为姿态模型存在已知错误。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """获取具有指定配置和权重的姿态估计模型

        参数:
            cfg (str | Path | dict, optional): 模型配置文件路径或字典
            weights (str | Path, optional): 模型权重文件的路径
            verbose (bool): 是否显示模型信息

        返回:
            (PoseModel): 初始化后的姿态估计模型
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """设置 PoseModel 的关键点形状属性"""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """返回用于验证的 PoseValidator 类实例"""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """获取数据集并确保其包含必需的 `kpt_shape` 键

        返回:
            (dict): 包含训练/验证/测试数据集和类别名称的字典

        异常:
            KeyError: 如果数据集中不存在 `kpt_shape` 键
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"在 {self.args.data} 中没有 `kpt_shape`。请参阅 https://docs.ultralytics.com/datasets/pose/")
        return data
