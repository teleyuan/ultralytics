from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
    """处理文本和视觉提示嵌入的 YOLOE 检测模型验证器类

    该类扩展了 DetectionValidator,为 YOLOE 模型提供专门的验证功能。它支持使用文本提示或从训练样本中提取的
    视觉提示嵌入进行验证,为基于提示的目标检测提供灵活的评估策略。

    属性:
        device (torch.device): 执行验证的设备
        args (namespace): 验证的配置参数
        dataloader (DataLoader): 验证数据的 DataLoader

    方法:
        get_visual_pe: 从训练样本中提取视觉提示嵌入
        preprocess: 预处理批次数据,确保视觉数据与图像在同一设备上
        get_vpe_dataloader: 为 LVIS 训练视觉提示样本创建数据加载器
        __call__: 使用文本或视觉提示嵌入运行验证

    示例:
        使用文本提示验证
        >>> validator = YOLOEDetectValidator()
        >>> stats = validator(model=model, load_vp=False)

        使用视觉提示验证
        >>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
        """从训练样本中提取视觉提示嵌入

        该方法处理数据加载器,使用 YOLOE 模型计算每个类别的视觉提示嵌入。它对嵌入进行归一化,
        并通过将其设置为零来处理不存在样本的类别。

        参数:
            dataloader (torch.utils.data.DataLoader): 提供训练样本的数据加载器
            model (YOLOEModel): 从中提取视觉提示嵌入的 YOLOE 模型

        返回:
            (torch.Tensor): 形状为 (1, num_classes, embed_dim) 的视觉提示嵌入
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "从样本中获取视觉提示嵌入"

        # 统计每个类别的样本数量
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # 提取视觉提示嵌入
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        # 对有样本的类别归一化嵌入,其他设置为零
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
        """为 LVIS 训练视觉提示样本创建数据加载器

        该方法使用指定的数据集准备用于视觉提示嵌入 (VPE) 的数据加载器。它将必要的变换(包括 LoadVisualPrompt)
        和配置应用于数据集以用于验证目的。

        参数:
            data (dict): 包含路径和设置的数据集配置字典

        返回:
            (torch.utils.data.DataLoader): 视觉提示样本的数据加载器
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )

    @smart_inference_mode()
    def __call__(
        self,
        trainer: Any | None = None,
        model: YOLOEModel | str | None = None,
        refer_data: str | None = None,
        load_vp: bool = False,
    ) -> dict[str, Any]:
        """使用文本或视觉提示嵌入在模型上运行验证

        该方法根据 load_vp 标志使用文本提示或视觉提示验证模型。它支持训练期间的验证(使用训练器对象)
        或使用提供的模型进行独立验证。对于视觉提示,可以指定参考数据以从不同数据集提取嵌入。

        参数:
            trainer (object, optional): 包含模型和设备的训练器对象
            model (YOLOEModel | str, optional): 要验证的模型。如果未提供训练器则必需
            refer_data (str, optional): 视觉提示的参考数据路径
            load_vp (bool): 是否加载视觉提示。如果为 False,则使用文本提示

        返回:
            (dict): 包含验证期间计算的指标的验证统计信息
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("使用视觉提示进行验证。")
                self.args.half = False
                # 直接使用相同的数据加载器提取训练期间的视觉嵌入
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("使用文本提示进行验证。")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "参考数据仅用于视觉提示验证。"
            self.device = select_device(self.args.device, verbose=False)

            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=self.device)  # model, ckpt
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if load_vp:
                LOGGER.info("使用视觉提示进行验证。")
                self.args.half = False
                # TODO: 需要检查参考数据中的类别名称是否与评估数据集一致
                # 可以使用相同的数据集或参考来提取视觉提示嵌入
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # 无提示
                return super().__call__(trainer, model)
            else:
                LOGGER.info("使用文本提示进行验证。")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """支持文本和视觉提示嵌入的 YOLOE 分割验证器"""

    pass
