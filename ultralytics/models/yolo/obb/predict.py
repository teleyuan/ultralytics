import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """扩展 DetectionPredictor 类的旋转边界框 (OBB) 预测器类

    该预测器处理旋转边界框检测任务,处理图像并返回带有旋转边界框的结果。

    属性:
        args (namespace): 预测器的配置参数
        model (torch.nn.Module): 加载的 YOLO OBB 模型

    示例:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb import OBBPredictor
        >>> args = dict(model="yolo11n-obb.pt", source=ASSETS)
        >>> predictor = OBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """使用可选的模型和数据配置覆盖初始化 OBBPredictor

        参数:
            cfg (dict, optional): 预测器的默认配置
            overrides (dict, optional): 优先于默认配置的配置覆盖
            _callbacks (list, optional): 预测期间调用的回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def construct_result(self, pred, img, orig_img, img_path):
        """从预测构建结果对象

        参数:
            pred (torch.Tensor): 预测的边界框、分数和旋转角度,形状为 (N, 7),其中
                最后一维包含 [x, y, w, h, confidence, class_id, angle]
            img (torch.Tensor): 预处理后的图像,形状为 (B, C, H, W)
            orig_img (np.ndarray): 预处理前的原始图像
            img_path (str): 原始图像的路径

        返回:
            (Results): 包含原始图像、图像路径、类别名称和旋转边界框的结果对象
        """
        rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)
