import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):
    """Ultralytics YOLO NAS 目标检测验证器。

    该类扩展了 Ultralytics 模型包中的 DetectionValidator，专门用于对 YOLO NAS 模型生成的
    原始预测结果进行后处理。它执行非极大值抑制以去除重叠和低置信度的边界框，最终生成最终检测结果。

    属性:
        args (Namespace): 包含后处理各种配置的命名空间，如置信度和 IoU 阈值。
        lb (torch.Tensor): 用于多标签 NMS 的可选张量。

    示例:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> validator = model.validator
        >>> # 假设 raw_preds 已可用
        >>> final_preds = validator.postprocess(raw_preds)

    注意:
        该类通常不会直接实例化，而是在 NAS 类内部使用。
    """

    def postprocess(self, preds_in):
        """对预测输出应用非极大值抑制。"""
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框格式从 xyxy 转换为 xywh
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 拼接边界框和分数并置换维度
        return super().postprocess(preds)
