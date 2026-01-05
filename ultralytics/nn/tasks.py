"""
YOLO 模型任务定义模块

这个模块是 Ultralytics YOLO 系列模型的核心架构定义文件。
它包含了所有 YOLO 模型的基类和特定任务的模型实现。

主要模型类:
    - BaseModel: 所有 YOLO 模型的抽象基类
    - DetectionModel: 目标检测模型（YOLO11, YOLOv8 等）
    - SegmentationModel: 实例分割模型（YOLO11-seg, YOLOv8-seg 等）
    - PoseModel: 姿态估计模型（YOLO11-pose, YOLOv8-pose 等）
    - ClassificationModel: 图像分类模型（YOLO11-cls 等）
    - OBBModel: 有向边界框检测模型（YOLO11-obb 等）
    - RTDETRDetectionModel: RT-DETR 实时检测模型
    - WorldModel: 开放词汇检测模型（YOLO-World）
    - YOLOEModel: 提示驱动检测模型（YOLOE）
    - YOLOESegModel: 提示驱动分割模型（YOLOE-Seg）

主要工具函数:
    - parse_model: 从 YAML 配置解析并构建模型
    - yaml_model_load: 加载 YAML 配置文件
    - torch_safe_load: 安全加载 PyTorch 权重
    - load_checkpoint: 加载模型检查点
    - guess_model_task: 推断模型任务类型
    - guess_model_scale: 推断模型规模（n/s/m/l/x）

模型架构特点:
    - 支持多尺度特征融合（FPN/PAN）
    - 模块化设计，易于扩展
    - 自动权重初始化
    - 支持模型融合优化
    - 支持多种损失函数
"""

# 导入标准库
import contextlib  # 上下文管理工具
import pickle  # Python 对象序列化
import re  # 正则表达式
import types  # Python 类型操作
from copy import deepcopy  # 深拷贝
from pathlib import Path  # 路径操作

# 导入 PyTorch 核心库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块

# 导入 Ultralytics 自动后端工具
from ultralytics.nn.autobackend import check_class_names  # 类别名称检查

# 导入 Ultralytics 神经网络模块
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Ultralytics 系列中所有 YOLO 模型的基类

    该类为 YOLO 模型提供通用功能,包括前向传播处理、模型融合、信息显示和权重加载能力。

    属性:
        model (torch.nn.Module): 神经网络模型
        save (list): 要保存输出的层索引列表
        stride (torch.Tensor): 模型步长值

    方法:
        forward: 执行训练或推理的前向传播
        predict: 对输入张量执行推理
        fuse: 融合 Conv2d 和 BatchNorm2d 层以进行优化
        info: 打印模型信息
        load: 将权重加载到模型中
        loss: 计算训练损失

    示例:
        创建 BaseModel 实例
        >>> model = BaseModel()
        >>> model.info()  # 显示模型信息
    """

    def forward(self, x, *args, **kwargs):
        """执行模型的前向传播以进行训练或推理

        如果 x 是字典,则计算并返回训练损失。否则,返回推理预测。

        参数:
            x (torch.Tensor | dict): 推理的输入张量,或包含图像张量和标签的训练字典
            *args (Any): 可变长度参数列表
            **kwargs (Any): 任意关键字参数

        返回:
            (torch.Tensor): 如果 x 是字典则返回损失 (训练),否则返回网络预测 (推理)
        """
        if isinstance(x, dict):  # 用于训练和训练期间验证的情况
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """通过网络执行前向传播

        参数:
            x (torch.Tensor): 模型的输入张量
            profile (bool): 如果为 True,则打印每层的计算时间
            visualize (bool): 如果为 True,则保存模型的特征图
            augment (bool): 在预测期间增强图像
            embed (list, optional): 要返回的特征向量/嵌入列表

        返回:
            (torch.Tensor): 模型的最后输出
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """通过网络执行前向传播

        参数:
            x (torch.Tensor): 模型的输入张量
            profile (bool): 如果为 True,则打印每层的计算时间
            visualize (bool): 如果为 True,则保存模型的特征图
            embed (list, optional): 要返回的特征向量/嵌入列表

        返回:
            (torch.Tensor): 模型的最后输出
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强推理结果"""
        LOGGER.warning(
            f"{self.__class__.__name__} 不支持 'augment=True' 预测。"
            f"恢复为单尺度预测。"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """分析模型单层在给定输入上的计算时间和 FLOPs

        参数:
            m (torch.nn.Module): 要分析的层
            x (torch.Tensor): 层的输入数据
            dt (list): 用于存储层计算时间的列表
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda 支持,无需安装 'ultralytics-thop'

        c = m == self.model[-1] and isinstance(x, list)  # 是否为最后一层列表,复制输入以进行原地修复
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """将模型的 Conv2d 和 BatchNorm2d 层融合为单层以提高计算效率

        返回:
            (torch.nn.Module): 返回融合后的模型
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # 移除 one2many 头
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """检查模型中的 BatchNorm 层数量是否少于某个阈值

        参数:
            thresh (int, optional): BatchNorm 层数量的阈值

        返回:
            (bool): 如果模型中的 BatchNorm 层数量少于阈值则返回 True,否则返回 False
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # 归一化层,即 BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # 如果模型中 BatchNorm 层少于 'thresh' 则返回 True

    def info(self, detailed=False, verbose=True, imgsz=640):
        """打印模型信息

        参数:
            detailed (bool): 如果为 True,则打印模型的详细信息
            verbose (bool): 如果为 True,则打印模型信息
            imgsz (int): 模型将要训练的图像大小
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """将函数应用于模型中所有非参数或已注册缓冲区的张量

        参数:
            fn (function): 要应用于模型的函数

        返回:
            (BaseModel): 更新后的 BaseModel 对象
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # 包括所有 Detect 子类,如 Segment、Pose、OBB、WorldDetect、YOLOEDetect、YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """将权重加载到模型中

        参数:
            weights (dict | torch.nn.Module): 要加载的预训练权重
            verbose (bool, optional): 是否记录迁移进度
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision 模型不是字典
        csd = model.float().state_dict()  # 检查点 state_dict,转换为 FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # 交集
        self.load_state_dict(updated_csd, strict=False)  # 加载
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # 目前硬编码为 yolo 模型
        # 主要用于增强多通道训练
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"从预训练权重迁移了 {len_updated_csd}/{len(self.model.state_dict())} 项")

    def loss(self, batch, preds=None):
        """计算损失

        参数:
            batch (dict): 要计算损失的批次
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """初始化 BaseModel 的损失准则"""
        raise NotImplementedError("compute_loss() 需要由任务头实现")


class DetectionModel(BaseModel):
    """YOLO 检测模型

    此类实现 YOLO 检测架构,处理目标检测任务的模型初始化、前向传播、增强推理和损失计算。

    属性:
        yaml (dict): 模型配置字典
        model (torch.nn.Sequential): 神经网络模型
        save (list): 要保存输出的层索引列表
        names (dict): 类别名称字典
        inplace (bool): 是否使用原地操作
        end2end (bool): 模型是否使用端到端检测
        stride (torch.Tensor): 模型步长值

    方法:
        __init__: 初始化 YOLO 检测模型
        _predict_augment: 执行增强推理
        _descale_pred: 反缩放增强推理后的预测
        _clip_augmented: 裁剪 YOLO 增强推理尾部
        init_criterion: 初始化损失准则

    示例:
        初始化检测模型
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLO 检测模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` 模块已弃用,改用 torch.nn.Identity。"
                "请删除本地 *.pt 文件并重新下载最新的模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        self.yaml["channels"] = ch  # 保存通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"使用 nc={nc} 覆盖 model.yaml nc={self.yaml['nc']}")
            self.yaml["nc"] = nc  # 覆盖 YAML 值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型、保存列表
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 构建步长
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有 Detect 子类,如 Segment、Pose、OBB、YOLOEDetect、YOLOESegment
            s = 256  # 2倍最小步长
            m.inplace = self.inplace

            def _forward(x):
                """通过模型执行前向传播,相应地处理不同的 Detect 子类类型"""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x)

            self.model.eval()  # 避免在训练开始前更改批统计
            m.training = True  # 设置为 True 以正确返回步长
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            self.model.train()  # 将模型设置回训练(默认)模式
            m.bias_init()  # 仅运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步长,例如 RTDETR

        # 初始化权重、偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强推理和训练输出

        参数:
            x (torch.Tensor): 输入图像张量

        返回:
            (torch.Tensor): 增强推理输出
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("模型不支持 'augment=True',恢复为单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度、宽度
        s = [1, 0.83, 0.67]  # 缩放比例
        f = [None, 3, None]  # 翻转 (2-上下, 3-左右)
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强尾部
        return torch.cat(y, -1), None  # 增强推理、训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """在增强推理后反缩放预测(逆操作)

        参数:
            p (torch.Tensor): 预测张量
            flips (int): 翻转类型 (0=无, 2=上下, 3=左右)
            scale (float): 缩放因子
            img_size (tuple): 原始图像大小 (高度, 宽度)
            dim (int): 分割的维度

        返回:
            (torch.Tensor): 反缩放后的预测
        """
        p[:, :4] /= scale  # 反缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反翻转上下
        elif flips == 3:
            x = img_size[1] - x  # 反翻转左右
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理尾部

        参数:
            y (list[torch.Tensor]): 检测张量列表

        返回:
            (list[torch.Tensor]): 裁剪后的检测张量
        """
        nl = self.model[-1].nl  # 检测层数量 (P3-P5)
        g = sum(4**x for x in range(nl))  # 网格点
        e = 1  # 排除层计数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大尺度
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小尺度
        return y

    def init_criterion(self):
        """初始化 DetectionModel 的损失准则"""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO 有向边界框 (OBB) 模型

    此类扩展 DetectionModel 以处理有向边界框检测任务,为旋转目标检测提供专门的损失计算。

    方法:
        __init__: 初始化 YOLO OBB 模型
        init_criterion: 初始化 OBB 检测的损失准则

    示例:
        初始化 OBB 模型
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLO OBB 模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化模型的损失准则"""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO 分割模型

    此类扩展 DetectionModel 以处理实例分割任务,为像素级目标检测和分割提供专门的损失计算。

    方法:
        __init__: 初始化 YOLO 分割模型
        init_criterion: 初始化分割的损失准则

    示例:
        初始化分割模型
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 Ultralytics YOLO 分割模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化 SegmentationModel 的损失准则"""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO 姿态模型

    此类扩展 DetectionModel 以处理人体姿态估计任务,为关键点检测和姿态估计提供专门的损失计算。

    属性:
        kpt_shape (tuple): 关键点数据的形状 (关键点数量, 维度数量)

    方法:
        __init__: 初始化 YOLO 姿态模型
        init_criterion: 初始化姿态估计的损失准则

    示例:
        初始化姿态模型
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """初始化 Ultralytics YOLO Pose 模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            data_kpt_shape (tuple): 关键点数据的形状
            verbose (bool): 是否显示模型信息
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # 加载模型 YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"使用 kpt_shape={data_kpt_shape} 覆盖 model.yaml kpt_shape={cfg['kpt_shape']}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """初始化 PoseModel 的损失准则"""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO 分类模型

    此类实现用于图像分类任务的 YOLO 分类架构,提供模型初始化、配置和输出重塑功能。

    属性:
        yaml (dict): 模型配置字典
        model (torch.nn.Sequential): 神经网络模型
        stride (torch.Tensor): 模型步长值
        names (dict): 类别名称字典

    方法:
        __init__: 初始化 ClassificationModel
        _from_yaml: 设置模型配置并定义架构
        reshape_outputs: 将模型更新为指定的类别数
        init_criterion: 初始化损失准则

    示例:
        初始化分类模型
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """使用 YAML、通道数、类别数、详细标志初始化 ClassificationModel

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """设置 Ultralytics YOLO 模型配置并定义模型架构

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典

        # 定义模型
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"使用 nc={nc} 覆盖 model.yaml nc={self.yaml['nc']}")
            self.yaml["nc"] = nc  # 覆盖 YAML 值
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc 未指定。必须在 model.yaml 或函数参数中指定 nc。")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型、保存列表
        self.stride = torch.Tensor([1])  # 无步长约束
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """如果需要,将 TorchVision 分类模型更新为类别数 'n'

        参数:
            model (torch.nn.Module): 要更新的模型
            nc (int): 新的类别数
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # 最后一个模块
        if isinstance(m, Classify):  # YOLO Classify() 头
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet、EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # 最后一个 torch.nn.Linear 索引
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # 最后一个 torch.nn.Conv2d 索引
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """初始化 ClassificationModel 的损失准则"""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) 检测模型类

    此类负责构建 RTDETR 架构、定义损失函数,并促进训练和推理过程。RTDETR 是一个目标检测和
    跟踪模型,它从 DetectionModel 基类扩展而来。

    属性:
        nc (int): 检测的类别数
        criterion (RTDETRDetectionLoss): 训练的损失函数

    方法:
        __init__: 初始化 RTDETRDetectionModel
        init_criterion: 初始化损失准则
        loss: 计算训练损失
        predict: 通过模型执行前向传播

    示例:
        初始化 RTDETR 模型
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """初始化 RTDETRDetectionModel

        参数:
            cfg (str | dict): 配置文件名或路径
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 初始化期间打印附加信息
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """将函数应用于模型中所有非参数或已注册缓冲区的张量

        参数:
            fn (function): 要应用于模型的函数

        返回:
            (RTDETRDetectionModel): 更新后的 BaseModel 对象
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """初始化 RTDETRDetectionModel 的损失准则"""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """计算给定数据批次的损失

        参数:
            batch (dict): 包含图像和标签数据的字典
            preds (torch.Tensor, optional): 预先计算的模型预测

        返回:
            loss_sum (torch.Tensor): 总损失值
            loss_items (torch.Tensor): 张量中的主要三个损失
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # 注意: 将 gt_bbox 和 gt_labels 预处理为列表
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # 注意: RTDETR 中有大约 12 个损失,用所有损失反向传播,但只显示主要的三个损失
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """通过模型执行前向传播

        参数:
            x (torch.Tensor): 输入张量
            profile (bool): 如果为 True,则分析每层的计算时间
            visualize (bool): 如果为 True,则保存特征图以供可视化
            batch (dict, optional): 用于评估的真实数据
            augment (bool): 如果为 True,则在推理期间执行数据增强
            embed (list, optional): 要返回的特征向量/嵌入列表

        返回:
            (torch.Tensor): 模型的输出张量
        """
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # 除头部部分外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # 头部推理
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World 模型

    此类实现用于开放词汇目标检测的 YOLOv8 World 模型,支持基于文本的类别规范和 CLIP 模型
    集成,以实现零样本检测功能。

    属性:
        txt_feats (torch.Tensor): 类别的文本特征嵌入
        clip_model (torch.nn.Module): 用于文本编码的 CLIP 模型

    方法:
        __init__: 初始化 YOLOv8 world 模型
        set_classes: 为离线推理设置类别
        get_text_pe: 获取文本位置嵌入
        predict: 使用文本特征执行前向传播
        loss: 使用文本特征计算损失

    示例:
        初始化 world 模型
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOv8 world 模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # 特征占位符
        self.clip_model = None  # CLIP 模型占位符
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """提前设置类别,以便模型可以在没有 clip 模型的情况下进行离线推理

        参数:
            text (list[str]): 类别名称列表
            batch (int): 处理文本标记的批次大小
            cache_clip_model (bool): 是否缓存 CLIP 模型
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """获取用于离线推理的文本位置嵌入,无需 CLIP 模型

        参数:
            text (list[str]): 类别名称列表
            batch (int): 处理文本标记的批次大小
            cache_clip_model (bool): 是否缓存 CLIP 模型

        返回:
            (torch.Tensor): 文本位置嵌入
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 为缺少 clip_model 属性的模型提供向后兼容性
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """通过模型执行前向传播

        参数:
            x (torch.Tensor): 输入张量
            profile (bool): 如果为 True,则分析每层的计算时间
            visualize (bool): 如果为 True,则保存特征图以供可视化
            txt_feats (torch.Tensor, optional): 文本特征,如果给定则使用它
            augment (bool): 如果为 True,则在推理期间执行数据增强
            embed (list, optional): 要返回的特征向量/嵌入列表

        返回:
            (torch.Tensor): 模型的输出张量
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # 输出
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # 除头部部分外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # 运行

            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失

        参数:
            batch (dict): 要计算损失的批次
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE 检测模型

    此类实现 YOLOE 架构,用于使用文本和视觉提示进行高效目标检测,支持基于提示和无提示的推理模式。

    属性:
        pe (torch.Tensor): 类别的提示嵌入
        clip_model (torch.nn.Module): 用于文本编码的 CLIP 模型

    方法:
        __init__: 初始化 YOLOE 模型
        get_text_pe: 获取文本位置嵌入
        get_visual_pe: 获取视觉嵌入
        set_vocab: 为无提示模型设置词汇表
        get_vocab: 获取融合的词汇表层
        set_classes: 为离线推理设置类别
        get_cls_pe: 获取类别位置嵌入
        predict: 使用提示执行前向传播
        loss: 使用提示计算损失

    示例:
        初始化 YOLOE 模型
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOE 模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """获取用于离线推理的文本位置嵌入,无需 CLIP 模型

        参数:
            text (list[str]): 类别名称列表
            batch (int): 处理文本标记的批次大小
            cache_clip_model (bool): 是否缓存 CLIP 模型
            without_reprta (bool): 是否返回未经 reprta 模块处理的文本嵌入

        返回:
            (torch.Tensor): 文本位置嵌入
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # 为缺少 clip_model 属性的模型提供向后兼容性
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # 运行辅助文本头

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """获取视觉嵌入

        参数:
            img (torch.Tensor): 输入图像张量
            visual (torch.Tensor): 视觉特征

        返回:
            (torch.Tensor): 视觉位置嵌入
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """为无提示模型设置词汇表

        参数:
            vocab (nn.ModuleList): 词汇表项列表
            names (list[str]): 类别名称列表
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # 为头缓存锚点
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # 预热

        # 无提示模型的重参数化
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """从模型获取融合的词汇表层

        参数:
            names (list): 类别名称列表

        返回:
            (nn.ModuleList): 词汇表模块列表
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # 将提示嵌入融合到分类头

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """提前设置类别,以便模型可以在没有 clip 模型的情况下进行离线推理

        参数:
            names (list[str]): 类别名称列表
            embeddings (torch.Tensor): 嵌入张量
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "无提示模型不支持设置类别。请尝试使用文本/视觉提示模型。"
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """获取类别位置嵌入

        参数:
            tpe (torch.Tensor, optional): 文本位置嵌入
            vpe (torch.Tensor, optional): 视觉位置嵌入

        返回:
            (torch.Tensor): 类别位置嵌入
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """通过模型执行前向传播

        参数:
            x (torch.Tensor): 输入张量
            profile (bool): 如果为 True,则分析每层的计算时间
            visualize (bool): 如果为 True,则保存特征图以供可视化
            tpe (torch.Tensor, optional): 文本位置嵌入
            augment (bool): 如果为 True,则在推理期间执行数据增强
            embed (list, optional): 要返回的特征向量/嵌入列表
            vpe (torch.Tensor, optional): 视觉位置嵌入
            return_vpe (bool): 如果为 True,则返回视觉位置嵌入

        返回:
            (torch.Tensor): 模型的输出张量
        """
        y, dt, embeddings = [], [], []  # 输出
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # 除头部部分外
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自更早的层
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # 运行

            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 展平
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """计算损失

        参数:
            batch (dict): 要计算损失的批次
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE 分割模型

    此类扩展 YOLOEModel 以使用文本和视觉提示处理实例分割任务,为像素级目标检测和分割提供专门的损失计算。

    方法:
        __init__: 初始化 YOLOE 分割模型
        loss: 使用提示计算分割损失

    示例:
        初始化 YOLOE 分割模型
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """使用给定的配置和参数初始化 YOLOE 分割模型

        参数:
            cfg (str | dict): 模型配置文件路径或字典
            ch (int): 输入通道数
            nc (int, optional): 类别数
            verbose (bool): 是否显示模型信息
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """计算损失

        参数:
            batch (dict): 要计算损失的批次
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """模型集成

    此类允许将多个 YOLO 模型组合成一个集成,通过模型平均或其他集成技术来提高性能。

    方法:
        __init__: 初始化模型集成
        forward: 从集成中的所有模型生成预测

    示例:
        创建模型集成
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """初始化模型集成"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """生成 YOLO 网络的最后一层

        参数:
            x (torch.Tensor): 输入张量
            augment (bool): 是否增强输入
            profile (bool): 是否分析模型
            visualize (bool): 是否可视化特征

        返回:
            y (torch.Tensor): 来自所有模型的连接预测
            train_out (None): 集成推理始终返回 None
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # 最大集成
        # y = torch.stack(y).mean(0)  # 平均集成
        y = torch.cat(y, 2)  # nms 集成, y 形状(B, HW, C)
        return y, None  # 推理、训练输出


# 函数 ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """用于临时添加或修改 Python 模块缓存 (sys.modules) 中模块的上下文管理器

    此函数可用于在运行时更改模块路径。当重构代码时很有用,您已将模块从一个位置移动到另一个位置,
    但仍希望支持旧的导入路径以实现向后兼容性。

    参数:
        modules (dict, optional): 将旧模块路径映射到新模块路径的字典
        attributes (dict, optional): 将旧模块属性映射到新模块属性的字典

    示例:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # 现在将导入 new.module
        >>> from old.module import attribute  # 现在将导入 new.module.attribute

    注意:
        更改仅在上下文管理器内有效,一旦上下文管理器退出就会撤销。
        请注意,直接操作 sys.modules 可能导致不可预测的结果,尤其是在较大的应用程序或库中。
        请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 在 sys.modules 中以旧名称设置属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 在 sys.modules 中以旧名称设置模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 移除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """在反序列化期间替换未知类的占位符类"""

    def __init__(self, *args, **kwargs):
        """初始化 SafeClass 实例,忽略所有参数"""
        pass

    def __call__(self, *args, **kwargs):
        """运行 SafeClass 实例,忽略所有参数"""
        pass


class SafeUnpickler(pickle.Unpickler):
    """自定义 Unpickler,用 SafeClass 替换未知类"""

    def find_class(self, module, name):
        """尝试查找类,如果不在安全模块中则返回 SafeClass

        参数:
            module (str): 模块名称
            name (str): 类名称

        返回:
            (type): 找到的类或 SafeClass
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # 添加其他被认为安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError,它会捕获错误,
    记录警告消息,并尝试通过 check_requirements() 函数安装缺失的模块。安装后,该函数再次尝试
    使用 torch.load() 加载模型。

    参数:
        weight (str): PyTorch 模型的文件路径
        safe_only (bool): 如果为 True,则在加载期间用 SafeClass 替换未知类

    返回:
        ckpt (dict): 加载的模型检查点
        file (str): 加载的文件名

    示例:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # 如果本地缺失则在线搜索
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # 通过自定义 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name 是缺失的模块名称
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"错误 ❌ {weight} 似乎是一个最初使用 https://github.com/ultralytics/yolov5 训练的 "
                    f"Ultralytics YOLOv5 模型。\n此模型与 https://github.com/ultralytics/ultralytics 的 "
                    f"YOLOv8 不向前兼容。"
                    f"\n建议的修复方法是使用最新的 'ultralytics' 包训练新模型或使用官方 Ultralytics 模型运行命令,"
                    f"例如 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"错误 ❌ {weight} 需要 numpy>=1.26.1,但安装的是 numpy=={__import__('numpy').__version__}。"
                )
            ) from e
        LOGGER.warning(
            f"{weight} 似乎需要 '{e.name}',但它不在 Ultralytics 要求中。"
            f"\n现在将为 '{e.name}' 运行自动安装,但此功能将来会被移除。"
            f"\n建议的修复方法是使用最新的 'ultralytics' 包训练新模型或使用官方 Ultralytics 模型运行命令,"
            f"例如 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # 安装缺失的模块
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # 文件可能是用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例
        LOGGER.warning(
            f"文件 '{weight}' 似乎保存不当或格式错误。"
            f"为获得最佳结果,请使用 model.save('filename.pt') 正确保存 YOLO 模型。"
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """加载单个模型权重

    参数:
        weight (str | Path): 模型权重路径
        device (torch.device, optional): 加载模型到的设备
        inplace (bool): 是否进行原地操作
        fuse (bool): 是否融合模型

    返回:
        model (torch.nn.Module): 加载的模型
        ckpt (dict): 模型检查点字典
    """
    ckpt, weight = torch_safe_load(weight)  # 加载检查点
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 合并模型和默认参数,优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 模型

    # 模型兼容性更新
    model.args = args  # 将参数附加到模型
    model.pt_path = weight  # 将 *.pt 文件路径附加到模型
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # 模型处于评估模式

    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 兼容性

    # 返回模型和检查点
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """将 YOLO model.yaml 字典解析为 PyTorch 模型

    参数:
        d (dict): 模型字典
        ch (int): 输入通道
        verbose (bool): 是否打印模型详细信息

    返回:
        model (torch.nn.Sequential): PyTorch 模型
        save (list): 输出层的排序列表
    """
    import ast

    # 参数
    legacy = True  # v3/v5/v8/v9 模型的向后兼容性
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"未传递模型规模。假设 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数,即 Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # 层、保存列表、输出通道
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # 具有 'repeat' 参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from、数量、模块、参数
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # 获取模块
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # 如果 c2 != nc (例如 Classify() 输出)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # 设置 1) 嵌入通道和 2) 头数
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # 重复次数
                n = 1
            if m is C3k2:  # 用于 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # 用于 L/X 尺寸
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # 特殊情况,通道参数必须在索引 1 处传递
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、'from' 索引、类型
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 附加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLOv8 模型

    参数:
        path (str | Path): YAML 文件的路径

    返回:
        (dict): 模型字典
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 模型现在使用 -p6 后缀。将 {path.stem} 重命名为 {new_stem}。")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 即 yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # 模型字典
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """从模型路径中提取模型规模的大小字符 n、s、m、l 或 x

    参数:
        model_path (str | Path): YOLO 模型 YAML 文件的路径

    返回:
        (str): 模型规模的大小字符 (n, s, m, l, 或 x)
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """从 PyTorch 模型的架构或配置推断模型任务

    参数:
        model (torch.nn.Module | dict): PyTorch 模型或 YAML 格式的模型配置

    返回:
        (str): 模型的任务 ('detect', 'segment', 'classify', 'pose', 'obb')
    """

    def cfg2task(cfg):
        """从 YAML 字典推断"""
        m = cfg["head"][-1][-2].lower()  # 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # 从模型配置推断
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # 从 PyTorch 模型推断
    if isinstance(model, torch.nn.Module):  # PyTorch 模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307: 安全评估已知属性路径
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: 安全评估已知属性路径
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # 从模型文件名推断
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # 无法从模型确定任务
    LOGGER.warning(
        "无法自动推断模型任务,假设 'task=detect'。"
        "明确定义模型的任务,即 'task=detect'、'segment'、'classify'、'pose' 或 'obb'。"
    )
    return "detect"  # 假设为检测
