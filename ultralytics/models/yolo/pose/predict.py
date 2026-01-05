from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PosePredictor(DetectionPredictor):
    """扩展 DetectionPredictor 类的姿态估计预测器类

    该类专门用于姿态估计任务,在继承 DetectionPredictor 的标准目标检测功能基础上,
    还能处理关键点检测。

    属性:
        args (namespace): 预测器的配置参数
        model (torch.nn.Module): 加载的 YOLO 姿态模型,具备关键点检测能力

    方法:
        construct_result: 从预测结果构建结果对象,包含关键点信息

    示例:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化用于姿态估计任务的 PosePredictor

        设置 PosePredictor 实例,将其配置为姿态检测任务,并处理 Apple MPS 的设备特定警告。

        参数:
            cfg (Any): 预测器的配置
            overrides (dict, optional): 优先于 cfg 的配置覆盖项
            _callbacks (list, optional): 预测期间调用的回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

    def construct_result(self, pred, img, orig_img, img_path):
        """从预测结果构建结果对象,包含关键点信息

        通过从预测结果中提取关键点数据并将其添加到结果对象中,扩展了父类的实现。

        参数:
            pred (torch.Tensor): 预测的边界框、分数和关键点,形状为 (N, 6+K*D),其中 N 是
                检测数量,K 是关键点数量,D 是关键点维度
            img (torch.Tensor): 预处理后的输入图像张量,形状为 (B, C, H, W)
            orig_img (np.ndarray): 预处理前的原始图像,numpy 数组格式
            img_path (str): 原始图像文件的路径

        返回:
            (Results): 包含原始图像、图像路径、类别名称、边界框和关键点的结果对象
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # 从预测中提取关键点并根据模型的关键点形状进行重塑
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # 将关键点坐标缩放到原始图像尺寸
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
