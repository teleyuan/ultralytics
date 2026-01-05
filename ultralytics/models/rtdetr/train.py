from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """百度开发的 RT-DETR 实时目标检测模型训练器类。

    该类扩展了 YOLO 的 DetectionTrainer 类，以适配 RT-DETR 的特定功能和架构。
    该模型利用 Vision Transformers，具有 IoU 感知查询选择和可调节推理速度等能力。

    属性:
        loss_names (tuple): 训练中使用的损失组件名称。
        data (dict): 包含类别数量和其他参数的数据集配置。
        args (dict): 训练参数和超参数。
        save_dir (Path): 保存训练结果的目录。
        test_loader (DataLoader): 用于验证/测试数据的 DataLoader。

    方法:
        get_model: 初始化并返回用于目标检测任务的 RT-DETR 模型。
        build_dataset: 构建并返回用于训练或验证的 RT-DETR 数据集。
        get_validator: 返回适用于 RT-DETR 模型验证的 DetectionValidator。

    示例:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    注意:
        - RT-DETR 中使用的 F.grid_sample 不支持 `deterministic=True` 参数。
        - AMP 训练可能导致 NaN 输出，并可能在二分图匹配期间产生错误。
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """初始化并返回用于目标检测任务的 RT-DETR 模型。

        参数:
            cfg (dict, optional): 模型配置。
            weights (str, optional): 预训练模型权重的路径。
            verbose (bool): 如果为 True，启用详细日志记录。

        返回:
            (RTDETRDetectionModel): 已初始化的模型。
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """构建并返回用于训练或验证的 RT-DETR 数据集。

        参数:
            img_path (str): 包含图像的文件夹路径。
            mode (str): 数据集模式，'train' 或 'val'。
            batch (int, optional): 矩形训练的批次大小。

        返回:
            (RTDETRDataset): 特定模式的数据集对象。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """返回适用于 RT-DETR 模型验证的 DetectionValidator。"""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
