import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch深度学习框架
from PIL import Image  # Python图像库，用于图像读取和处理

from ultralytics.data.augment import classify_transforms  # 导入分类任务的数据增强转换函数
from ultralytics.engine.predictor import BasePredictor  # 导入预测器基类
from ultralytics.engine.results import Results  # 导入结果封装类
from ultralytics.utils import DEFAULT_CFG, ops  # 导入默认配置和操作工具函数


class ClassificationPredictor(BasePredictor):
    """
    分类预测器类 - 基于分类模型进行预测的预测器。

    该预测器处理分类模型的特定需求，包括图像预处理和后处理预测结果以生成分类结果。

    Attributes:
        args (dict): 预测器的配置参数字典。

    Methods:
        preprocess: 将输入图像转换为模型兼容的格式。
        postprocess: 将模型预测结果处理为Results对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.classify import ClassificationPredictor
        >>> args = dict(model="yolo11n-cls.pt", source=ASSETS)
        >>> predictor = ClassificationPredictor(overrides=args)
        >>> predictor.predict_cli()

    Notes:
        - Torchvision分类模型也可以传递给'model'参数，例如 model='resnet18'。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化分类预测器并设置任务类型为'classify'。

        该构造函数初始化一个ClassificationPredictor实例，它扩展了BasePredictor用于分类任务。
        无论输入配置如何，它都会确保任务设置为'classify'。
      
        Args:
            cfg (dict): 包含预测设置的默认配置字典。
            overrides (dict, optional): 覆盖cfg的配置项，优先级高于cfg。
            _callbacks (list, optional): 在预测过程中执行的回调函数列表。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"  # 强制设置任务类型为分类

    def setup_source(self, source):
        """
        设置数据源、推理模式和分类转换。
        """
        super().setup_source(source)
        # 检查模型的转换尺寸是否需要更新
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        # 如果需要更新或不是PyTorch模型，则创建新的转换；否则使用模型自带的转换
        self.transforms = (
            classify_transforms(self.imgsz) if updated or not self.model.pt else self.model.model.transforms
        )

    def preprocess(self, img):
        """
        将输入图像转换为模型兼容的张量格式并进行适当的归一化。
        """
        if not isinstance(img, torch.Tensor):
            # 将BGR图像转换为RGB，然后应用transforms，最后堆叠成批次
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        # 将图像移动到模型所在的设备（CPU或GPU）
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # 根据模型精度转换为fp16或fp32

    def postprocess(self, preds, img, orig_imgs):
        """
        处理预测结果，返回包含分类概率的Results对象。

        Args:
            preds (torch.Tensor): 模型的原始预测结果。
            img (torch.Tensor): 预处理后的输入图像。
            orig_imgs (list[np.ndarray] | torch.Tensor): 预处理前的原始图像。
        Returns:
            (list[Results]): 包含每个图像分类结果的Results对象列表。
        """
        # 如果原始图像不是列表（而是torch.Tensor），则转换为numpy数组并反转颜色通道
        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        # 如果预测结果是列表或元组，取第一个元素
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        # 为每个预测结果创建Results对象
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
