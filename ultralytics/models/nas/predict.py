import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class NASPredictor(DetectionPredictor):
    """Ultralytics YOLO NAS 目标检测预测器。

    该类扩展了 Ultralytics 引擎的 DetectionPredictor，负责对 YOLO NAS 模型生成的原始
    预测结果进行后处理。它应用非极大值抑制等操作，并将边界框缩放到原始图像尺寸。

    属性:
        args (Namespace): 包含后处理各种配置的命名空间，包括置信度阈值、IoU 阈值、
            类别无关 NMS 标志、最大检测数量和类别过滤选项。
        model (torch.nn.Module): 用于推理的 YOLO NAS 模型。
        batch (list): 待处理的输入批次。

    示例:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> predictor = model.predictor

        假设 raw_preds、img、orig_imgs 已可用
        >>> results = predictor.postprocess(raw_preds, img, orig_imgs)

    注意:
        通常情况下，该类不会直接实例化。它在 NAS 类内部使用。
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """对 NAS 模型预测结果进行后处理以生成最终检测结果。

        该方法接收来自 YOLO NAS 模型的原始预测结果，转换边界框格式，并应用后处理操作
        以生成与 Ultralytics 结果可视化和分析工具兼容的最终检测结果。

        参数:
            preds_in (list): 来自 NAS 模型的原始预测结果，通常包含边界框和类别分数。
            img (torch.Tensor): 输入到模型的图像张量，形状为 (B, C, H, W)。
            orig_imgs (list | torch.Tensor | np.ndarray): 预处理前的原始图像，用于将坐标
                缩放回原始尺寸。

        返回:
            (list): 包含批次中每张图像处理后预测结果的 Results 对象列表。

        示例:
            >>> predictor = NAS("yolo_nas_s").predictor
            >>> results = predictor.postprocess(raw_preds, img, orig_imgs)
        """
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框从 xyxy 格式转换为 xywh 格式
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 将边界框与类别分数拼接
        return super().postprocess(preds, img, orig_imgs)
