from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """扩展 DetectionPredictor 类的分割模型预测器类

    该类专门用于处理分割模型输出,在预测结果中同时处理边界框和掩码。

    属性:
        args (dict): 预测器的配置参数
        model (torch.nn.Module): 加载的 YOLO 分割模型
        batch (list): 当前正在处理的图像批次

    方法:
        postprocess: 应用非极大值抑制并处理分割检测
        construct_results: 从预测结果构建结果对象列表
        construct_result: 从预测结果构建单个结果对象

    示例:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化 SegmentationPredictor,设置配置、覆盖项和回调函数

        该类专门用于处理分割模型输出,在预测结果中同时处理边界框和掩码。

        参数:
            cfg (dict): 预测器的配置
            overrides (dict, optional): 优先于 cfg 的配置覆盖项
            _callbacks (list, optional): 预测期间调用的回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """对输入批次中的每张图像应用非极大值抑制并处理分割检测

        参数:
            preds (tuple): 模型预测,包含边界框、分数、类别和掩码系数
            img (torch.Tensor): 模型格式的输入图像张量,形状为 (B, C, H, W)
            orig_imgs (list | torch.Tensor | np.ndarray): 原始图像或图像批次

        返回:
            (list): 包含批次中每张图像的分割预测的 Results 对象列表。每个 Results 对象包含边界框和
                分割掩码

        示例:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo11n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # 提取原型 - 如果是 PyTorch 模型则为元组,如果是导出模型则为数组
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """从预测结果构建结果对象列表

        参数:
            preds (list[torch.Tensor]): 预测的边界框、分数和掩码列表
            img (torch.Tensor): 预处理后的图像
            orig_imgs (list[np.ndarray]): 预处理前的原始图像列表
            protos (list[torch.Tensor]): 原型掩码列表

        返回:
            (list[Results]): 包含原始图像、图像路径、类别名称、边界框和掩码的结果对象列表
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """从预测结果构建单个结果对象

        参数:
            pred (torch.Tensor): 预测的边界框、分数和掩码
            img (torch.Tensor): 预处理后的图像
            orig_img (np.ndarray): 预处理前的原始图像
            img_path (str): 原始图像的路径
            proto (torch.Tensor): 原型掩码

        返回:
            (Results): 包含原始图像、图像路径、类别名称、边界框和掩码的结果对象
        """
        if pred.shape[0] == 0:  # 保存空边界框
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # 只保留有掩码的预测
            if not all(keep):  # 大多数预测有掩码
                pred, masks = pred[keep], masks[keep]  # 索引较慢
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
