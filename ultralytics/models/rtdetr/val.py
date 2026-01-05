from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # 元组或列表


class RTDETRDataset(YOLODataset):
    """RT-DETR（实时检测与跟踪）数据集类，扩展基础 YOLODataset 类。

    该专用数据集类专为 RT-DETR 目标检测模型设计，针对实时检测和跟踪任务进行了优化。

    属性:
        augment (bool): 是否应用数据增强。
        rect (bool): 是否使用矩形训练。
        use_segments (bool): 是否使用分割掩码。
        use_keypoints (bool): 是否使用关键点标注。
        imgsz (int): 训练的目标图像尺寸。

    方法:
        load_image: 从数据集索引加载一张图像。
        build_transforms: 构建数据集的变换管道。

    示例:
        初始化 RT-DETR 数据集
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """通过继承 YOLODataset 类初始化 RTDETRDataset 类。

        该构造函数设置专门为 RT-DETR（实时检测与跟踪）模型优化的数据集，
        基于 YOLODataset 的基础功能构建。

        参数:
            *args (Any): 传递给父类 YOLODataset 的可变长度参数列表。
            data (dict | None): 包含数据集信息的字典。如果为 None，将使用默认值。
            **kwargs (Any): 传递给父类 YOLODataset 的额外关键字参数。
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """从数据集索引 'i' 加载一张图像。

        参数:
            i (int): 要加载的图像索引。
            rect_mode (bool, optional): 是否使用矩形模式进行批量推理。

        返回:
            im (np.ndarray): 加载的图像，以 NumPy 数组形式。
            hw_original (tuple[int, int]): 原始图像尺寸，格式为 (高度, 宽度)。
            hw_resized (tuple[int, int]): 调整后的图像尺寸，格式为 (高度, 宽度)。

        示例:
            从数据集加载图像
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """构建数据集的变换管道。

        参数:
            hyp (dict, optional): 变换的超参数。

        返回:
            (Compose): 变换函数的组合。
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])  # 空变换列表
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """RTDETRValidator 扩展 DetectionValidator 类，为 RT-DETR（实时 DETR）目标检测模型提供专门定制的验证功能。

    该类允许构建用于验证的 RTDETR 专用数据集，应用非极大值抑制进行后处理，
    并相应地更新评估指标。

    属性:
        args (Namespace): 验证的配置参数。
        data (dict): 数据集配置字典。

    方法:
        build_dataset: 构建用于验证的 RTDETR 数据集。
        postprocess: 对预测输出应用非极大值抑制。

    示例:
        初始化并运行 RT-DETR 验证
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    注意:
        有关属性和方法的更多详细信息，请参阅父类 DetectionValidator。
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建 RTDETR 数据集。

        参数:
            img_path (str): 包含图像的文件夹路径。
            mode (str, optional): `train` 模式或 `val` 模式，用户可以为每种模式自定义不同的增强。
            batch (int, optional): 批次大小，用于 `rect` 模式。

        返回:
            (RTDETRDataset): 配置用于 RT-DETR 验证的数据集。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # 不使用增强
            hyp=self.args,
            rect=False,  # 不使用矩形模式
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测结果缩放到原始图像尺寸。"""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """对预测输出应用非极大值抑制。

        参数:
            preds (torch.Tensor | list | tuple): 来自模型的原始预测。如果是张量，形状应为
                (batch_size, num_predictions, num_classes + 4)，其中最后一个维度包含边界框坐标和类别分数。

        返回:
            (list[dict[str, torch.Tensor]]): 每张图像的字典列表，每个字典包含:
                - 'bboxes': 形状为 (N, 4) 的边界框坐标张量
                - 'conf': 形状为 (N,) 的置信度分数张量
                - 'cls': 形状为 (N,) 的类别索引张量
        """
        if not isinstance(preds, (list, tuple)):  # PyTorch 推理返回列表，导出推理返回 list[0] 张量
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # 过滤
            # 按置信度排序以正确获得内部指标
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果序列化为 COCO JSON 格式。

        参数:
            predn (dict[str, torch.Tensor]): 预测字典，包含 'bboxes'、'conf' 和 'cls' 键，分别对应
                边界框坐标、置信度分数和类别预测。
            pbatch (dict[str, Any]): 批次字典，包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file'。
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # 原始空间预测
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # 原始空间预测
        box = ops.xyxy2xywh(box)  # 转换为 xywh 格式
        box[:, :2] -= box[:, 2:] / 2  # 从中心点坐标转换为左上角坐标
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
