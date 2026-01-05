from __future__ import annotations  # 允许在类型注解中使用类自身

from pathlib import Path  # 用于处理文件路径的面向对象接口
from typing import Any  # 用于类型注解的Any类型

import torch  # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式训练模块

from ultralytics.data import ClassificationDataset, build_dataloader  # 导入分类数据集和数据加载器构建函数
from ultralytics.engine.validator import BaseValidator  # 导入验证器基类
from ultralytics.utils import LOGGER, RANK  # 导入日志记录器和分布式训练进程等级
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix  # 导入分类指标和混淆矩阵类
from ultralytics.utils.plotting import plot_images  # 导入图像绘制函数


class ClassificationValidator(BaseValidator):
    """用于基于分类模型进行验证的验证器类，扩展 BaseValidator。

    该验证器处理分类模型的验证过程，包括指标计算、混淆矩阵生成和结果可视化。

    属性:
        targets (list[torch.Tensor]): 真实的类别标签。
        pred (list[torch.Tensor]): 模型预测结果。
        metrics (ClassifyMetrics): 用于计算和存储分类指标的对象。
        names (dict): 类别索引到类别名称的映射。
        nc (int): 类别数量。
        confusion_matrix (ConfusionMatrix): 用于评估模型在各类别上性能的混淆矩阵。

    方法:
        get_desc: 返回总结分类指标的格式化字符串。
        init_metrics: 初始化混淆矩阵、类别名称和跟踪容器。
        preprocess: 通过将数据移动到设备来预处理输入批次。
        update_metrics: 使用模型预测和批次目标更新运行指标。
        finalize_metrics: 完成指标计算，包括混淆矩阵和处理速度。
        postprocess: 从模型输出中提取主要预测。
        get_stats: 计算并返回指标字典。
        build_dataset: 创建用于验证的 ClassificationDataset 实例。
        get_dataloader: 构建并返回用于分类验证的数据加载器。
        print_results: 打印分类模型的评估指标。
        plot_val_samples: 绘制带有真实标签的验证图像样本。
        plot_predictions: 绘制带有预测类别标签的图像。

    示例:
        >>> from ultralytics.models.yolo.classify import ClassificationValidator
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10")
        >>> validator = ClassificationValidator(args=args)
        >>> validator()

    注意:
        Torchvision 分类模型也可以传递给 'model' 参数，例如 model='resnet18'。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """使用数据加载器、保存目录和其他参数初始化分类验证器。

        参数:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (str | Path, optional): 保存结果的目录。
            args (dict, optional): 包含模型和验证配置的参数。
            _callbacks (list, optional): 验证过程中调用的回调函数列表。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None  # 初始化目标标签容器
        self.pred = None  # 初始化预测结果容器
        self.args.task = "classify"  # 强制设置任务类型为分类
        self.metrics = ClassifyMetrics()  # 初始化分类指标计算对象

    def get_desc(self) -> str:
        """返回总结分类指标的格式化字符串。"""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化混淆矩阵、类别名称和用于预测与目标的跟踪容器。"""
        self.names = model.names  # 从模型获取类别名称
        self.nc = len(model.names)  # 设置类别数量
        self.pred = []  # 初始化预测结果列表
        self.targets = []  # 初始化目标标签列表
        self.confusion_matrix = ConfusionMatrix(names=model.names)  # 初始化混淆矩阵

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """通过将数据移动到设备并转换为适当的数据类型来预处理输入批次。"""
        # 将图像移动到指定设备（CPU或GPU），CUDA设备使用非阻塞传输
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        # 根据配置转换为half精度（fp16）或float精度（fp32）
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        # 将类别标签移动到指定设备
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """使用模型预测和批次目标更新运行指标。

        参数:
            preds (torch.Tensor): 模型预测结果，通常是每个类别的 logits 或概率。
            batch (dict): 包含图像和类别标签的批次数据。

        注意:
            该方法将 top-N 预测（按置信度降序排列）追加到预测列表中以供后续评估。
            N 限制为 5 和类别数量的最小值。
        """
        n5 = min(len(self.names), 5)  # 计算top-N的N值，最多为5
        # 对预测结果排序并取top-N，转换为int32类型并移到CPU
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        # 将目标标签转换为int32类型并移到CPU
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self) -> None:
        """完成指标计算，包括混淆矩阵和处理速度。

        示例:
            >>> validator = ClassificationValidator()
            >>> validator.pred = [torch.tensor([[0, 1, 2]])]  # 一个样本的 Top-3 预测
            >>> validator.targets = [torch.tensor([0])]  # 真实类别
            >>> validator.finalize_metrics()
            >>> print(validator.metrics.confusion_matrix)  # 访问混淆矩阵

        注意:
            该方法处理累积的预测和目标以生成混淆矩阵，可选择绘制它，并使用速度信息更新指标对象。
        """
        # 处理分类预测以生成混淆矩阵
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            # 绘制归一化和非归一化的混淆矩阵
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        # 将处理速度、保存目录和混淆矩阵信息添加到指标对象
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]) -> torch.Tensor:
        """从模型输出中提取主要预测（如果输出是列表或元组格式）。"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self) -> dict[str, float]:
        """通过处理目标和预测来计算并返回指标字典。"""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息（用于分布式训练）。"""
        if RANK == 0:  # 如果是主进程
            # 创建容器用于收集所有进程的预测和目标
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            # 从所有进程收集数据到主进程
            dist.gather_object(self.pred, gathered_preds, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            # 合并所有进程的预测和目标
            self.pred = [pred for rank in gathered_preds for pred in rank]
            self.targets = [targets for rank in gathered_targets for targets in rank]
        elif RANK > 0:  # 如果是工作进程
            # 将数据发送到主进程
            dist.gather_object(self.pred, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def build_dataset(self, img_path: str) -> ClassificationDataset:
        """创建用于验证的 ClassificationDataset 实例。"""
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path: Path | str, batch_size: int) -> torch.utils.data.DataLoader:
        """构建并返回用于分类验证的数据加载器。

        参数:
            dataset_path (str | Path): 数据集目录的路径。
            batch_size (int): 每批样本数量。

        返回:
            (torch.utils.data.DataLoader): 用于分类验证数据集的 DataLoader 对象。
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self) -> None:
        """打印分类模型的评估指标。"""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制带有真实标签的验证图像样本。

        参数:
            batch (dict[str, Any]): 包含批次数据的字典，包含 'img'（图像）和 'cls'（类别标签）。
            ni (int): 用于命名输出文件的批次索引。

        示例:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
            >>> validator.plot_val_samples(batch, 0)
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # 添加批次索引用于绘图
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """绘制带有预测类别标签的图像并保存可视化结果。

        参数:
            batch (dict[str, Any]): 包含图像和其他信息的批次数据。
            preds (torch.Tensor): 模型预测结果，形状为 (batch_size, num_classes)。
            ni (int): 用于命名输出文件的批次索引。

        示例:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224)}
            >>> preds = torch.rand(16, 10)  # 16 张图像，10 个类别
            >>> validator.plot_predictions(batch, preds, 0)
        """
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0]),
            cls=torch.argmax(preds, dim=1),
            conf=torch.amax(preds, dim=1),
        )
        plot_images(
            batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测
