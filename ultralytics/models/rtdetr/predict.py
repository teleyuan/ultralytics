import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """RT-DETR（实时检测 Transformer）预测器，扩展 BasePredictor 类进行预测。

    该类利用 Vision Transformers 提供实时目标检测，同时保持高精度。
    支持高效混合编码和 IoU 感知查询选择等关键特性。

    属性:
        imgsz (int): 推理的图像尺寸（必须是正方形且缩放填充）。
        args (dict): 预测器的参数覆盖。
        model (torch.nn.Module): 加载的 RT-DETR 模型。
        batch (list): 当前处理的输入批次。

    方法:
        postprocess: 后处理原始模型预测以生成边界框和置信度分数。
        pre_transform: 在将输入图像送入模型进行推理之前进行预变换。

    示例:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """后处理模型的原始预测以生成边界框和置信度分数。

        该方法根据 `self.args` 中指定的置信度和类别过滤检测结果。
        它将模型预测转换为包含正确缩放边界框的 Results 对象。

        参数:
            preds (list | tuple): 来自模型的 [predictions, extra] 列表，其中 predictions 包含边界框和分数。
            img (torch.Tensor): 处理后的输入图像，形状为 (N, 3, H, W)。
            orig_imgs (list | torch.Tensor): 原始的未处理图像。

        返回:
            results (list[Results]): 包含后处理边界框、置信度分数和类别标签的 Results 对象列表。
        """
        if not isinstance(preds, (list, tuple)):  # PyTorch 推理返回列表，导出推理返回 list[0] 张量
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而非列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # 过滤
            pred = pred[pred[:, 4].argsort(descending=True)][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # 将 x 坐标缩放到原始宽度
            pred[..., [1, 3]] *= oh  # 将 y 坐标缩放到原始高度
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """在将输入图像送入模型进行推理之前进行预变换。

        输入图像经过 letterbox 处理以确保正方形宽高比并进行缩放填充。

        参数:
            im (list[np.ndarray] | torch.Tensor): 输入图像，张量形状为 (N, 3, H, W)，
                列表形式为 [(H, W, 3) x N]。

        返回:
            (list): 已预变换的图像列表，准备进行模型推理。
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
