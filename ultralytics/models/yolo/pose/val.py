from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """扩展 DetectionValidator 类的姿态估计验证器类

    该验证器专门用于姿态估计任务,处理关键点并实现姿态评估的专用指标。

    属性:
        sigma (np.ndarray): 用于 OKS 计算的 Sigma 值,可以是 OKS_SIGMA 或 1/关键点数量
        kpt_shape (list[int]): 关键点的形状,对于 COCO 格式通常为 [17, 3]
        args (dict): 验证器的参数,包括设置为 "pose" 的任务
        metrics (PoseMetrics): 用于姿态评估的指标对象

    方法:
        preprocess: 通过将关键点数据转换为浮点数并移动到设备来预处理批次
        get_desc: 返回字符串格式的评估指标描述
        init_metrics: 初始化 YOLO 模型的姿态估计指标
        _prepare_batch: 通过将关键点转换为浮点数并缩放到原始尺寸来准备批次进行处理
        _prepare_pred: 准备和缩放预测中的关键点以进行姿态处理
        _process_batch: 通过计算检测和真实值之间的交并比 (IoU) 返回正确预测矩阵
        plot_val_samples: 绘制并保存带有真实边界框和关键点的验证集样本
        plot_predictions: 绘制并保存带有边界框和关键点的模型预测
        save_one_txt: 将 YOLO 姿态检测保存到归一化坐标的文本文件中
        pred_to_json: 将 YOLO 预测转换为 COCO JSON 格式
        eval_json: 使用 COCO JSON 格式评估目标检测模型

    示例:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    注意:
        该类使用姿态特定功能扩展了 DetectionValidator。它使用 OKS 计算的 sigma 值初始化,
        并设置 PoseMetrics 进行评估。使用 Apple MPS 时会显示警告,因为姿态模型存在已知错误。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """初始化用于姿态估计验证的 PoseValidator 对象

        该验证器专门用于姿态估计任务,处理关键点并实现姿态评估的专用指标。

        参数:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器
            save_dir (Path | str, optional): 保存结果的目录
            args (dict, optional): 验证器的参数,包括设置为 "pose" 的任务
            _callbacks (list, optional): 验证期间执行的回调函数列表
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """通过将关键点数据转换为浮点数并移动到设备来预处理批次"""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """返回字符串格式的评估指标描述"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化 YOLO 姿态验证的评估指标

        参数:
            model (torch.nn.Module): 要验证的模型
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> dict[str, torch.Tensor]:
        """后处理 YOLO 预测以提取和重塑姿态估计的关键点

        该方法通过从预测的 'extra' 字段中提取关键点并根据关键点形状配置重塑它们来扩展父类的后处理。
        关键点从扁平格式重塑为正确的维度结构(对于 COCO 姿态格式通常为 [N, 17, 3])。

        参数:
            preds (torch.Tensor): YOLO 姿态模型的原始预测张量,包含边界框、置信度分数、
                类别预测和关键点数据

        返回:
            (dict[torch.Tensor]): 处理后的预测字典,每个包含:
                - 'bboxes': 边界框坐标
                - 'conf': 置信度分数
                - 'cls': 类别预测
                - 'keypoints': 形状为 (-1, *self.kpt_shape) 的重塑关键点坐标

        注意:
            如果预测中不存在关键点(空关键点),则跳过该预测并继续下一个。关键点从 'extra' 字段中提取,
            该字段包含超出基本检测的额外任务特定数据。
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # 如果存在则移除 extra
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """通过将关键点转换为浮点数并缩放到原始尺寸来准备批次进行处理

        参数:
            si (int): 批次索引
            batch (dict[str, Any]): 包含批次数据的字典,具有 'keypoints'、'batch_idx' 等键

        返回:
            (dict[str, Any]): 关键点缩放到原始图像尺寸的准备好的批次

        注意:
            该方法通过添加关键点处理来扩展父类的 _prepare_batch 方法。
            关键点从归一化坐标缩放到原始图像尺寸。
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """通过计算检测和真实值之间的交并比 (IoU) 返回正确预测矩阵

        参数:
            preds (dict[str, torch.Tensor]): 包含预测数据的字典,具有 'cls' 键表示类别预测,
                'keypoints' 键表示关键点预测
            batch (dict[str, Any]): 包含真实数据的字典,具有 'cls' 键表示类别标签,'bboxes'
                键表示边界框,'keypoints' 键表示关键点标注

        返回:
            (dict[str, np.ndarray]): 包含正确预测矩阵的字典,包括 'tp_p' 表示在 10 个 IoU 级别上的
                姿态真正例

        注意:
            面积计算中使用的 `0.53` 缩放因子引用自
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` 来自 https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # 使用关键点 IoU 更新 tp
        return tp

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """将 YOLO 姿态检测保存到归一化坐标的文本文件中

        参数:
            predn (dict[str, torch.Tensor]): 预测字典,包含 'bboxes'、'conf'、'cls' 和 'keypoints' 键
            save_conf (bool): 是否保存置信度分数
            shape (tuple[int, int]): 原始图像的形状 (高度, 宽度)
            file (Path): 保存检测结果的输出文件路径

        注意:
            输出格式为: 类别ID x中心 y中心 宽度 高度 置信度 关键点,其中关键点是每个点的
            归一化 (x, y, 可见性) 值。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测转换为 COCO JSON 格式

        该方法接收预测张量和文件名,将边界框从 YOLO 格式转换为 COCO 格式,
        并将结果追加到内部 JSON 字典 (self.jdict) 中。

        参数:
            predn (dict[str, torch.Tensor]): 预测字典,包含 'bboxes'、'conf'、'cls' 和 'keypoints'
                张量
            pbatch (dict[str, Any]): 批次字典,包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file'

        注意:
            该方法从文件名主干中提取图像 ID(如果是数字则为整数,否则为字符串),
            将边界框从 xyxy 格式转换为 xywh 格式,并在保存到 JSON 字典之前将坐标从中心调整到左上角。
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # 关键点

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸"""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """使用 COCO JSON 格式评估目标检测模型"""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # 标注文件
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
