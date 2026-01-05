"""
RT-DETR 模型接口模块

RT-DETR (Real-Time DEtection TRansformer) 是百度开发的基于 Vision Transformer 的实时目标检测器。
它在保持 Transformer 架构优势的同时，实现了与 YOLO 相媲美的实时性能。

核心特性:
    - 实时检测性能（与 YOLO 速度相当）
    - 基于 Transformer 的端到端架构
    - 无需 NMS 后处理
    - 高效混合编码器（CNN + Transformer）
    - IoU 感知的查询选择机制
    - 可调节的推理速度

技术亮点:
    - AIFI (Attention-based Intrascale Feature Interaction): 基于注意力的尺度内特征交互
    - CCFM (Cross-scale Feature Fusion Module): 跨尺度特征融合模块
    - Uncertainty-minimal Query Selection: 最小化不确定性的查询选择
    - 支持 TensorRT 加速，在 CUDA 等加速后端表现出色

参考文献:
    https://arxiv.org/pdf/2304.08069.pdf
"""

# 导入必要的模块
from ultralytics.engine.model import Model  # 基础模型类
from ultralytics.nn.tasks import RTDETRDetectionModel  # RT-DETR 检测模型
from ultralytics.utils.torch_utils import TORCH_1_11  # PyTorch 版本检查

# 导入 RT-DETR 组件
from .predict import RTDETRPredictor  # RT-DETR 预测器
from .train import RTDETRTrainer  # RT-DETR 训练器
from .val import RTDETRValidator  # RT-DETR 验证器


class RTDETR(Model):
    """RT-DETR 模型接口类，基于 Vision Transformer 的实时目标检测器。

    该类提供了百度 RT-DETR 模型的统一接口，结合了 Transformer 的高精度和实时检测的速度。
    支持高效混合编码、IoU 感知查询选择和可调节的推理速度。

    属性:
        model (str): 预训练模型的路径

    方法:
        task_map: 返回 RT-DETR 的任务映射，关联任务与对应的 Ultralytics 类

    示例:
        使用预训练模型初始化 RT-DETR
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")  # 加载 RT-DETR Large 模型
        >>> results = model("image.jpg")  # 执行检测
        >>> results = model.train(data="coco.yaml", epochs=100)  # 训练模型
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """使用给定的预训练模型文件初始化 RT-DETR 模型。

        参数:
            model (str): 预训练模型的路径，支持 .pt、.yaml 和 .yml 格式
                可用模型: rtdetr-l.pt (Large), rtdetr-x.pt (XLarge)

        异常:
            AssertionError: 如果 PyTorch 版本低于 1.11
        """
        # RT-DETR 需要 PyTorch 1.11 或更高版本
        assert TORCH_1_11, "RTDETR 需要 torch>=1.11"
        # 调用父类初始化，设置任务类型为检测
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """返回 RT-DETR 的任务映射，关联任务与对应的 Ultralytics 类。

        该属性定义了 RT-DETR 模型支持的任务及其对应的处理类。
        包括预测器、验证器、训练器和模型架构。

        返回:
            (dict): 任务名称到 Ultralytics 任务类的映射字典
                - predictor: RTDETRPredictor - 预测器类
                - validator: RTDETRValidator - 验证器类
                - trainer: RTDETRTrainer - 训练器类
                - model: RTDETRDetectionModel - 模型架构类
        """
        return {
            "detect": {  # 检测任务
                "predictor": RTDETRPredictor,  # 预测器
                "validator": RTDETRValidator,  # 验证器
                "trainer": RTDETRTrainer,  # 训练器
                "model": RTDETRDetectionModel,  # 模型架构
            }
        }
