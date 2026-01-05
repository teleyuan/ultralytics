"""
YOLO 目标检测验证模块

该模块实现了 YOLO 目标检测模型的验证功能,包括:
    - mAP 计算: mAP@0.5, mAP@0.5:0.95 等指标
    - 精度召回率计算: Precision, Recall, F1-Score
    - 混淆矩阵生成: 可视化模型的分类混淆情况
    - COCO/LVIS 格式支持: 支持标准数据集评估
    - 结果可视化: 绘制预测结果和标签

主要类:
    - DetectionValidator: 目标检测验证器,继承自 BaseValidator

验证流程:
    1. 构建验证数据集和数据加载器
    2. 模型前向传播
    3. 后处理 (NMS)
    4. 计算 IoU 并匹配预测和真值
    5. 计算 mAP 等指标
    6. 生成混淆矩阵和可视化

评估指标:
    - mAP@0.5: IoU 阈值为 0.5 时的平均精度均值
    - mAP@0.5:0.95: IoU 阈值从 0.5 到 0.95 的平均精度均值
    - Precision: 精确率
    - Recall: 召回率
"""

from __future__ import annotations  # 启用延迟类型注解评估

import os  # 操作系统接口
from pathlib import Path  # 路径操作
from typing import Any  # 类型提示

import numpy as np  # 数值计算
import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # 分布式训练

# 导入数据处理和验证相关模块
from ultralytics.data import build_dataloader, build_yolo_dataset, converter  # 数据加载和转换
from ultralytics.engine.validator import BaseValidator  # 验证器基类
from ultralytics.utils import LOGGER, RANK, nms, ops  # 日志、进程rank、NMS、操作工具
from ultralytics.utils.checks import check_requirements  # 依赖检查
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou  # 评估指标
from ultralytics.utils.plotting import plot_images  # 可视化工具


class DetectionValidator(BaseValidator):
    """扩展 BaseValidator 类的目标检测验证器类

    该类实现了目标检测任务的特定验证功能,包括指标计算、预测处理和结果可视化。

    属性:
        is_coco (bool): 数据集是否为 COCO
        is_lvis (bool): 数据集是否为 LVIS
        class_map (list[int]): 从模型类别索引到数据集类别索引的映射
        metrics (DetMetrics): 目标检测指标计算器
        iouv (torch.Tensor): 用于 mAP 计算的 IoU 阈值
        niou (int): IoU 阈值数量
        lb (list[Any]): 用于存储混合保存的真实标签列表
        jdict (list[dict[str, Any]]): 用于存储 JSON 检测结果的列表
        stats (dict[str, list[torch.Tensor]]): 验证期间存储统计信息的字典

    示例:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """使用必要的变量和设置初始化检测验证器。

        参数:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (Path, optional): 保存结果的目录。
            args (dict[str, Any], optional): 验证器的参数。
            _callbacks (list[Any], optional): 回调函数列表。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # 用于 mAP@0.5:0.95 计算的 IoU 向量
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """对 YOLO 验证的批次图像进行预处理。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次数据。

        返回:
            (dict[str, Any]): 预处理后的批次数据。
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化 YOLO 检测验证的评估指标。

        参数:
            model (torch.nn.Module): 要验证的模型。
        """
        val = self.data.get(self.args.split, "")  # 验证路径
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # 是否为 COCO 数据集
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # 是否为 LVIS 数据集
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # 运行最终验证
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """返回 YOLO 模型类别指标的格式化字符串摘要。"""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """对预测输出应用非极大值抑制。

        参数:
            preds (torch.Tensor): 模型的原始预测结果。

        返回:
            (list[dict[str, torch.Tensor]]): NMS 处理后的预测结果,每个字典包含 'bboxes'、'conf'、
                'cls' 和 'extra' 张量。
        """
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """为验证准备一批图像和标注。

        参数:
            si (int): 批次索引。
            batch (dict[str, Any]): 包含图像和标注的批次数据。

        返回:
            (dict[str, Any]): 处理后标注的准备好的批次数据。
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # 目标框
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """准备预测结果以便与真实标签进行评估。

        参数:
            pred (dict[str, torch.Tensor]): 模型后处理的预测结果。

        返回:
            (dict[str, torch.Tensor]): 原始空间中的准备好的预测结果。
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """使用新的预测结果和真实标签更新指标。

        参数:
            preds (list[dict[str, torch.Tensor]]): 模型的预测结果列表。
            batch (dict[str, Any]): 包含真实标签的批次数据。
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # 评估
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # 保存结果
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """设置指标速度和混淆矩阵的最终值。"""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in self.metrics.stats.keys()}
            for stats_dict in gathered_stats:
                for key in merged_stats:
                    merged_stats[key].extend(stats_dict[key])
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.metrics.stats = merged_stats
            self.seen = len(self.dataloader.dataset)  # 数据集中的总图像数
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []
            self.metrics.clear_stats()

    def get_stats(self) -> dict[str, Any]:
        """计算并返回指标统计信息。

        返回:
            (dict[str, Any]): 包含指标结果的字典。
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """打印每个类别的训练/验证集指标。"""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        # 打印每个类别的结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """返回正确预测矩阵。

        参数:
            preds (dict[str, torch.Tensor]): 包含预测数据的字典,具有 'bboxes' 和 'cls' 键。
            batch (dict[str, Any]): 包含真实标签数据的批次字典,具有 'bboxes' 和 'cls' 键。

        返回:
            (dict[str, np.ndarray]): 包含 'tp' 键的字典,值为形状为 (N, 10) 的正确预测矩阵,
                对应 10 个 IoU 阈值级别。
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """构建 YOLO 数据集。

        参数:
            img_path (str): 包含图像的文件夹路径。
            mode (str): `train` 模式或 `val` 模式,用户可以为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小,用于 `rect` 模式。

        返回:
            (Dataset): YOLO 数据集。
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """构建并返回数据加载器。

        参数:
            dataset_path (str): 数据集路径。
            batch_size (int): 每个批次的大小。

        返回:
            (torch.utils.data.DataLoader): 用于验证的数据加载器。
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制验证图像样本。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次数据。
            ni (int): 批次索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """在输入图像上绘制预测的边界框并保存结果。

        参数:
            batch (dict[str, Any]): 包含图像和标注的批次数据。
            preds (list[dict[str, torch.Tensor]]): 模型的预测结果列表。
            ni (int): 批次索引。
            max_det (Optional[int]): 要绘制的最大检测数量。
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # 向预测结果添加批次索引
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # 转换为 xywh 格式
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测结果

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """以特定格式将 YOLO 检测结果保存到 txt 文件中,使用归一化坐标。

        参数:
            predn (dict[str, torch.Tensor]): 包含预测结果的字典,具有 'bboxes'、'conf' 和 'cls' 键。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像的形状 (高度, 宽度)。
            file (Path): 保存检测结果的文件路径。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果序列化为 COCO json 格式。

        参数:
            predn (dict[str, torch.Tensor]): 包含预测结果的字典,具有 'bboxes'、'conf' 和 'cls' 键,
                包含边界框坐标、置信度分数和类别预测。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。

        示例:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy 中心点转换为左上角坐标
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """评估 JSON 格式的 YOLO 输出并返回性能统计信息。

        参数:
            stats (dict[str, Any]): 当前的统计信息字典。

        返回:
            (dict[str, Any]): 更新后的包含 COCO/LVIS 评估结果的统计信息字典。
        """
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # 标注文件
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """使用 faster-coco-eval 库评估 COCO/LVIS 指标。

        使用 faster-coco-eval 库执行评估,计算目标检测的 mAP 指标。使用计算的指标更新
        提供的统计信息字典,包括 mAP50、mAP50-95,以及 LVIS 特定指标(如适用)。

        参数:
            stats (dict[str, Any]): 用于存储计算的指标和统计信息的字典。
            pred_json (str | Path): 包含 COCO 格式预测结果的 JSON 文件路径。
            anno_json (str | Path): 包含 COCO 格式真实标注的 JSON 文件路径。
            iou_types (str | list[str]): 用于评估的 IoU 类型。可以是单个字符串或字符串列表。常见值
                包括 "bbox"、"segm"、"keypoints"。默认为 "bbox"。
            suffix (str | list[str]): 附加到统计信息字典中指标名称的后缀。如果提供多种类型,
                应与 iou_types 对应。默认为 "Box"。

        返回:
            (dict[str, Any]): 更新后的包含计算的 COCO/LVIS 评估指标的统计信息字典。
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 要评估的图像
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # 更新 mAP50-95 和 mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # 始终使用 box mAP50-95 作为 fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
