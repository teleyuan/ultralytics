"""
神经网络模块块

这个模块包含了 YOLO 模型中使用的各种神经网络构建块。
这些块是构建深度学习模型的基础组件，包括卷积块、注意力机制、特征融合模块等。

主要模块类别:
    - 基础块: DFL, Proto, SPP, SPPF, C1-C3 系列
    - Bottleneck 变体: Bottleneck, BottleneckCSP, GhostBottleneck, RepBottleneck
    - 注意力机制: Attention, PSA, C2PSA, C2fPSA, ImagePoolingAttn, AAttn
    - ResNet 组件: ResNetBlock, ResNetLayer
    - CSP 变体: RepCSP, RepNCSPELAN4, ELAN1
    - 下采样: AConv, ADown, SCDown
    - 对比学习: ContrastiveHead, BNContrastiveHead
    - 特殊模块: CIB, SAVPE, SwiGLUFFN, TorchVision
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) 的积分模块

    该模块在论文 Generalized Focal Loss 中提出
    参考: https://ieeexplore.ieee.org/document/9792391

    DFL 通过学习边界框坐标的分布来提高目标检测精度，而不是直接回归单一值。
    """

    def __init__(self, c1: int = 16):
        """初始化具有给定输入通道数的卷积层

        参数:
            c1 (int): 输入通道数，默认为 16
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 DFL 模块应用于输入张量并返回转换后的输出"""
        b, _, a = x.shape  # batch, channels, anchors (批次大小, 通道数, 锚点数)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO 模型用于分割任务的掩码原型 (Proto) 模块

    该模块用于生成分割掩码的原型向量，这些原型与检测结果结合生成最终的实例分割掩码。
    """

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """初始化 Ultralytics YOLO 模型的掩码原型模块

        参数:
            c1 (int): 输入通道数
            c_ (int): 中间层通道数，默认为 256
            c2 (int): 输出通道数（原型数量），默认为 32
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用上采样的输入图像执行前向传播"""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """PPHGNetV2 的 Stem 块，包含 5 个卷积层和 1 个最大池化层

    该模块是 PPHGNetV2 网络的入口块，用于特征提取的初始阶段。
    参考: https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """初始化 PPHGNetV2 的 StemBlock

        参数:
            c1 (int): 输入通道数
            cm (int): 中间层通道数
            c2 (int): 输出通道数
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PPHGNetV2 主干层的前向传播"""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """PPHGNetV2 的 HG 块，包含 2 个卷积和 LightConv

    该模块是 PPHGNetV2 网络的核心构建块，使用轻量级卷积和压缩-激励机制提高特征提取效率。
    参考: https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """使用指定参数初始化 HGBlock

        参数:
            c1 (int): 输入通道数
            cm (int): 中间层通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小，默认为 3
            n (int): LightConv 或 Conv 块的数量，默认为 6
            lightconv (bool): 是否使用 LightConv，默认为 False
            shortcut (bool): 是否使用跳跃连接，默认为 False
            act (nn.Module): 激活函数，默认为 ReLU
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv (压缩卷积)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv (激励卷积)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PPHGNetV2 主干层的前向传播"""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """空间金字塔池化 (Spatial Pyramid Pooling, SPP) 层

    该模块通过多尺度池化提取不同感受野的特征，增强模型对多尺度目标的检测能力。
    参考: https://arxiv.org/abs/1406.4729
    """

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """使用输入/输出通道和池化核大小初始化 SPP 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (tuple): 最大池化的卷积核大小元组，默认为 (5, 9, 13)
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SPP 层的前向传播，执行空间金字塔池化"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """空间金字塔池化 - 快速版 (Spatial Pyramid Pooling - Fast, SPPF) 层

    由 Glenn Jocher 为 YOLOv5 设计的快速 SPP 层，通过串行池化代替并行池化提高计算效率。
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        """使用给定的输入/输出通道和卷积核大小初始化 SPPF 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小，默认为 5

        注意:
            该模块等效于 SPP(k=(5, 9, 13))，但计算效率更高
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入应用连续池化操作并返回拼接的特征图"""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """包含 1 个卷积的 CSP Bottleneck 块

    该模块是 CSP (Cross Stage Partial) 结构的简化版本，使用单个卷积层和残差连接。
    """

    def __init__(self, c1: int, c2: int, n: int = 1):
        """初始化包含 1 个卷积的 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): 卷积层数量，默认为 1
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用卷积和残差连接"""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """包含 2 个卷积的 CSP Bottleneck 块

    该模块使用 CSP 结构和两个卷积层，通过分离特征流来减少计算量并增强梯度流。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 2 个卷积的 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2) (可选激活函数)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention() (或空间注意力)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过包含 2 个卷积的 CSP Bottleneck 执行前向传播"""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """包含 2 个卷积的 CSP Bottleneck 的更快实现

    该模块是 C2 的改进版本，通过优化特征拼接方式提高推理速度，常用于 YOLOv8 等模型。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """初始化包含 2 个卷积的 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 False
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2) (可选激活函数)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 C2f 层执行前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """使用 split() 代替 chunk() 执行前向传播"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """包含 3 个卷积的 CSP Bottleneck 块

    该模块使用三个卷积层构建 CSP 结构，通过双路径特征提取提高模型性能。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 3 个卷积的 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2) (可选激活函数)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过包含 3 个卷积的 CSP Bottleneck 执行前向传播"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """包含交叉卷积的 C3 模块

    该模块在 C3 基础上使用交叉卷积（1x3 和 3x1），减少参数量的同时保持感受野。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含交叉卷积的 C3 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """使用重参数化卷积的 C3 模块

    该模块使用 RepConv（重参数化卷积）构建 CSP Bottleneck，训练时使用多分支结构，推理时融合为单个卷积提高速度。
    """

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """初始化包含单个卷积的 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): RepConv 块的数量，默认为 3
            e (float): 扩展比率，默认为 1.0
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RepC3 模块的前向传播"""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """包含 Transformer 块的 C3 模块

    该模块将 Bottleneck 替换为 TransformerBlock，引入自注意力机制来捕获长距离依赖关系。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 TransformerBlock 的 C3 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Transformer 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """包含 GhostBottleneck 的 C3 模块

    该模块使用 Ghost Bottleneck 代替标准 Bottleneck，通过廉价操作生成冗余特征图来减少计算量。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化包含 GhostBottleneck 的 C3 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Ghost Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck 模块

    该模块来自华为 Noah 方舟实验室的高效 AI 主干网络。
    参考: https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """初始化 Ghost Bottleneck 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小，默认为 3
            s (int): 步长，默认为 1
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw (逐点卷积)
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw (深度卷积)
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear (线性逐点卷积)
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用跳跃连接和拼接"""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """标准 Bottleneck 模块

    该模块是深度学习中常用的瓶颈块，通过降维-卷积-升维的方式减少计算量。
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """初始化标准 Bottleneck 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            k (tuple): 卷积核大小元组，默认为 (3, 3)
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用带可选跳跃连接的 Bottleneck"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck 模块

    该模块实现了跨阶段部分网络 (Cross Stage Partial Networks)。
    参考: https://github.com/WongKinYiu/CrossStagePartialNetworks
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化 CSP Bottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3) (应用于 cv2 和 cv3 的拼接结果)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用包含 3 个卷积的 CSP Bottleneck"""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """包含标准卷积层的 ResNet 块

    该模块实现了 ResNet 的基本构建块，使用 1x1-3x3-1x1 卷积结构。
    """

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """初始化 ResNet 块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            s (int): 步长，默认为 1
            e (int): 扩展比率，默认为 4
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 ResNet 块执行前向传播"""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """包含多个 ResNet 块的 ResNet 层

    该模块将多个 ResNet 块组合成一个层，可作为 ResNet 网络的阶段。
    """

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """初始化 ResNet 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            s (int): 步长，默认为 1
            is_first (bool): 是否为第一层，默认为 False
            n (int): ResNet 块的数量，默认为 1
            e (int): 扩展比率，默认为 4
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 ResNet 层执行前向传播"""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid 注意力块

    该模块使用最大值和 Sigmoid 激活实现注意力机制，通过引导特征调节输入特征。
    """

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """初始化 MaxSigmoidAttnBlock

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            nh (int): 注意力头数，默认为 1
            ec (int): 嵌入通道数，默认为 128
            gc (int): 引导通道数，默认为 512
            scale (bool): 是否使用可学习的缩放参数，默认为 False
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """MaxSigmoidAttnBlock 的前向传播

        参数:
            x (torch.Tensor): 输入张量
            guide (torch.Tensor): 引导张量，用于调节注意力

        返回:
            (torch.Tensor): 应用注意力后的输出张量
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """包含额外注意力模块的 C2f 模块

    该模块在 C2f 基础上添加了 MaxSigmoidAttnBlock，引入引导注意力机制增强特征表达。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """初始化包含注意力机制的 C2f 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck 块的数量，默认为 1
            ec (int): 注意力的嵌入通道数，默认为 128
            nh (int): 注意力头数，默认为 1
            gc (int): 注意力的引导通道数，默认为 512
            shortcut (bool): 是否使用跳跃连接，默认为 False
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2) (可选激活函数)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """通过包含注意力的 C2f 层执行前向传播

        参数:
            x (torch.Tensor): 输入张量
            guide (torch.Tensor): 用于注意力的引导张量

        返回:
            (torch.Tensor): 处理后的输出张量
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """使用 split() 代替 chunk() 执行前向传播

        参数:
            x (torch.Tensor): 输入张量
            guide (torch.Tensor): 用于注意力的引导张量

        返回:
            (torch.Tensor): 处理后的输出张量
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """图像池化注意力模块

    该模块使用图像感知信息增强文本嵌入，通过多尺度池化特征与文本特征交互。
    用于视觉-语言任务中的特征融合。
    """

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """初始化 ImagePoolingAttn 模块

        参数:
            ec (int): 嵌入通道数，默认为 256
            ch (tuple): 特征图的通道维度元组，默认为空
            ct (int): 文本嵌入的通道维度，默认为 512
            nh (int): 注意力头数，默认为 8
            k (int): 池化核大小，默认为 3
            scale (bool): 是否使用可学习的缩放参数，默认为 False
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """ImagePoolingAttn 的前向传播

        参数:
            x (list[torch.Tensor]): 输入特征图列表
            text (torch.Tensor): 文本嵌入

        返回:
            (torch.Tensor): 增强后的文本嵌入
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """对比学习头模块

    该模块实现了视觉-语言模型中区域-文本相似度的对比学习头，用于计算图像区域与文本的匹配度。
    """

    def __init__(self):
        """初始化 ContrastiveHead，设置区域-文本相似度参数"""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses (注意：使用 -10.0 保持初始分类损失与其他损失的一致性)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """对比学习的前向函数

        参数:
            x (torch.Tensor): 图像特征
            w (torch.Tensor): 文本特征

        返回:
            (torch.Tensor): 相似度分数
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """使用批归一化的对比学习头

    该模块使用批归一化代替 L2 归一化实现对比学习，提高训练稳定性。

    参数:
        embed_dims (int): 文本和图像特征的嵌入维度
    """

    def __init__(self, embed_dims: int):
        """初始化 BNContrastiveHead

        参数:
            embed_dims (int): 特征的嵌入维度
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses (注意：使用 -10.0 保持初始分类损失与其他损失的一致性)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable (使用 -1.0 更稳定)
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """融合 BNContrastiveHead 模块中的批归一化层"""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """直接传递输入，不做改变"""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """使用批归一化的对比学习前向函数

        参数:
            x (torch.Tensor): 图像特征
            w (torch.Tensor): 文本特征

        返回:
            (torch.Tensor): 相似度分数
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """重参数化 Bottleneck 模块

    该模块使用 RepConv 实现 Bottleneck，结合重参数化技术提高推理效率。
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """初始化 RepBottleneck

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            k (tuple): 卷积核大小元组，默认为 (3, 3)
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """可重复跨阶段部分网络 (Repeatable Cross Stage Partial Network) 模块

    该模块用于高效特征提取，使用 RepBottleneck 构建 CSP 结构。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """初始化 RepCSP 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): RepBottleneck 块的数量，默认为 1
            shortcut (bool): 是否使用跳跃连接，默认为 True
            g (int): 卷积分组数，默认为 1
            e (float): 扩展比率，默认为 0.5
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels (隐藏通道数)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN 模块

    该模块结合了 RepCSP 和 ELAN (Efficient Layer Aggregation Network) 架构，用于高效特征聚合。
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """初始化 CSP-ELAN 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            c3 (int): 中间层通道数
            c4 (int): RepCSP 的中间层通道数
            n (int): RepCSP 块的数量，默认为 1
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 RepNCSPELAN4 层执行前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """使用 split() 代替 chunk() 执行前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """包含 4 个卷积的 ELAN1 模块

    该模块是 ELAN 架构的简化版本，使用标准卷积代替 RepCSP。
    """

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """初始化 ELAN1 层

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            c3 (int): 中间层通道数
            c4 (int): 卷积的中间层通道数
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv 模块，使用平均池化和卷积实现下采样

    该模块结合了平均池化和卷积操作，用于特征图的下采样。
    """

    def __init__(self, c1: int, c2: int):
        """初始化 AConv 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 AConv 层执行前向传播"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown 下采样模块

    该模块使用平均池化、最大池化和卷积的组合实现特征图下采样。
    """

    def __init__(self, c1: int, c2: int):
        """初始化 ADown 模块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 ADown 层执行前向传播"""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN 模块

    该模块结合了 SPP (空间金字塔池化) 和 ELAN 架构，通过多尺度池化提取丰富的特征表示。
    """

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """初始化 SPP-ELAN 块

        参数:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            c3 (int): 中间层通道数
            k (int): 最大池化的卷积核大小，默认为 5
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 SPPELAN 层执行前向传播"""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear 模块

    该模块实现了通道分支线性变换，将输入特征映射到多个输出分支。
    """

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """初始化 CBLinear 模块

        参数:
            c1 (int): 输入通道数
            c2s (list[int]): 输出通道数列表
            k (int): 卷积核大小，默认为 1
            s (int): 步长，默认为 1
            p (int | None): 填充，默认为 None
            g (int): 分组数，默认为 1
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """通过 CBLinear 层执行前向传播"""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse 模块,用于融合多个特征图。"""

    def __init__(self, idx: list[int]):
        """初始化 CBFuse 模块。

        参数:
            idx (list[int]): 用于特征选择的索引列表。
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """执行通过 CBFuse 层的前向传播。

        参数:
            xs (list[torch.Tensor]): 输入张量列表。

        返回:
            (torch.Tensor): 融合后的输出张量。
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """带有 2 个卷积的 CSP Bottleneck 快速实现。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """初始化带有两个卷积的 CSP 瓶颈层。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块的数量。
            shortcut (bool): 是否使用快捷连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行通过 C3f 层的前向传播。"""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """带有 2 个卷积的 CSP Bottleneck 快速实现。"""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """初始化 C3k2 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 块的数量。
            c3k (bool): 是否使用 C3k 块。
            e (float): 扩展比例。
            g (int): 卷积分组数。
            shortcut (bool): 是否使用快捷连接。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k 是一个带有可自定义卷积核大小的 CSP 瓶颈模块,用于神经网络中的特征提取。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """初始化 C3k 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck 块的数量。
            shortcut (bool): 是否使用快捷连接。
            g (int): 卷积分组数。
            e (float): 扩展比例。
            k (int): 卷积核大小。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW 是表示 RepVGG 架构中的深度可分离卷积块的类。"""

    def __init__(self, ed: int) -> None:
        """初始化 RepVGGDW 模块。

        参数:
            ed (int): 输入和输出通道数。
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RepVGGDW 块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 应用深度可分离卷积后的输出张量。
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """执行 RepVGGDW 块的融合前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 应用深度可分离卷积后的输出张量。
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """融合 RepVGGDW 块中的卷积层。

        此方法融合卷积层并相应地更新权重和偏置。
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """紧凑倒置块(Compact Inverted Block, CIB)模块。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        shortcut (bool, optional): 是否添加快捷连接。默认为 True。
        e (float, optional): 隐藏通道的缩放因子。默认为 0.5。
        lk (bool, optional): 是否在第三个卷积层使用 RepVGGDW。默认为 False。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """初始化 CIB 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用快捷连接。
            e (float): 扩展比例。
            lk (bool): 是否使用 RepVGGDW。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 CIB 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 输出张量。
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB 类表示带有 C2f 和 CIB 模块的卷积块。

    参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, optional): 堆叠的 CIB 模块数量。默认为 1。
        shortcut (bool, optional): 是否使用快捷连接。默认为 False。
        lk (bool, optional): 是否使用大卷积核。默认为 False。
        g (int, optional): 分组卷积的组数。默认为 1。
        e (float, optional): CIB 模块的扩展比例。默认为 0.5。
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """初始化 C2fCIB 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): CIB 模块的数量。
            shortcut (bool): 是否使用快捷连接。
            lk (bool): 是否使用大卷积核。
            g (int): 卷积分组数。
            e (float): 扩展比例。
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """对输入张量执行自注意力的注意力模块。

    参数:
        dim (int): 输入张量的维度。
        num_heads (int): 注意力头的数量。
        attn_ratio (float): 注意力键维度与头维度的比率。

    属性:
        num_heads (int): 注意力头的数量。
        head_dim (int): 每个注意力头的维度。
        key_dim (int): 注意力键的维度。
        scale (float): 注意力分数的缩放因子。
        qkv (Conv): 用于计算查询、键和值的卷积层。
        proj (Conv): 用于投影注意力值的卷积层。
        pe (Conv): 用于位置编码的卷积层。
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """初始化多头注意力模块。

        参数:
            dim (int): 输入维度。
            num_heads (int): 注意力头的数量。
            attn_ratio (float): 键维度的注意力比率。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行注意力模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 自注意力后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock 类实现用于神经网络的位置敏感注意力块。

    该类封装了应用多头注意力和前馈神经网络层的功能,支持可选的快捷连接。

    属性:
        attn (Attention): 多头注意力模块。
        ffn (nn.Sequential): 前馈神经网络模块。
        add (bool): 指示是否添加快捷连接的标志。

    方法:
        forward: 执行通过 PSABlock 的前向传播,应用注意力和前馈层。

    示例:
        创建 PSABlock 并执行前向传播
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """初始化 PSABlock。

        参数:
            c (int): 输入和输出通道数。
            attn_ratio (float): 键维度的注意力比率。
            num_heads (int): 注意力头的数量。
            shortcut (bool): 是否使用快捷连接。
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行通过 PSABlock 的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA 类用于在神经网络中实现位置敏感注意力。

    该类封装了对输入张量应用位置敏感注意力和前馈网络的功能,增强特征提取和处理能力。

    属性:
        c (int): 应用初始卷积后的隐藏通道数。
        cv1 (Conv): 1x1 卷积层,将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 卷积层,将输出通道数减少到 c。
        attn (Attention): 用于位置敏感注意力的注意力模块。
        ffn (nn.Sequential): 用于进一步处理的前馈网络。

    方法:
        forward: 对输入张量应用位置敏感注意力和前馈网络。

    示例:
        创建 PSA 模块并应用到输入张量
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """初始化 PSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            e (float): 扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 PSA 模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 经过注意力和前馈处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA 模块,带有用于增强特征提取和处理的注意力机制。

    该模块实现了带有注意力机制的卷积块,以增强特征提取和处理能力。它包括一系列用于自注意力和前馈操作的 PSABlock 模块。

    属性:
        c (int): 隐藏通道数。
        cv1 (Conv): 1x1 卷积层,将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 卷积层,将输出通道数减少到 c。
        m (nn.Sequential): PSABlock 模块的顺序容器,用于注意力和前馈操作。

    方法:
        forward: 执行通过 C2PSA 模块的前向传播,应用注意力和前馈操作。

    示例:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    注意:
        该模块本质上与 PSA 模块相同,但经过重构以允许堆叠更多 PSABlock 模块。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2PSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块的数量。
            e (float): 扩展比例。
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过一系列 PSA 块处理输入张量。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 处理后的输出张量。
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA 模块,使用 PSA 块增强特征提取。

    该类通过结合 PSA 块来扩展 C2f 模块,以改进注意力机制和特征提取。

    属性:
        c (int): 隐藏通道数。
        cv1 (Conv): 1x1 卷积层,将输入通道数减少到 2*c。
        cv2 (Conv): 1x1 卷积层,将输出通道数减少到 c。
        m (nn.ModuleList): 用于特征提取的 PSA 块列表。

    方法:
        forward: 执行通过 C2fPSA 模块的前向传播。
        forward_split: 使用 split() 而不是 chunk() 执行前向传播。

    示例:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """初始化 C2fPSA 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock 模块的数量。
            e (float): 扩展比例。
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown 模块,使用可分离卷积进行下采样。

    该模块使用逐点卷积和深度卷积的组合执行下采样,有助于在保持通道信息的同时高效地减小输入张量的空间维度。

    属性:
        cv1 (Conv): 逐点卷积层,减少通道数。
        cv2 (Conv): 深度卷积层,执行空间下采样。

    方法:
        forward: 对输入张量应用 SCDown 模块。

    示例:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """初始化 SCDown 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 步长。
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入张量应用卷积和下采样。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 下采样后的输出张量。
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision 模块,允许加载任何 torchvision 模型。

    该类提供了一种从 torchvision 库加载模型的方法,可选地加载预训练权重,并通过截断或展开层来自定义模型。

    参数:
        model (str): 要加载的 torchvision 模型名称。
        weights (str, optional): 要加载的预训练权重。默认为 "DEFAULT"。
        unwrap (bool, optional): 将模型展开为包含除最后 `truncate` 层之外的所有层的顺序容器。
        truncate (int, optional): 如果 `unwrap` 为 True,从末尾截断的层数。默认为 2。
        split (bool, optional): 将中间子模块的输出作为列表返回。默认为 False。

    属性:
        m (nn.Module): 加载的 torchvision 模型,可能已截断和展开。
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """从 torchvision 加载模型和权重。

        参数:
            model (str): 要加载的 torchvision 模型名称。
            weights (str): 要加载的预训练权重。
            unwrap (bool): 是否展开模型。
            truncate (int): 要截断的层数。
            split (bool): 是否拆分输出。
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行通过模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor | list[torch.Tensor]): 输出张量或张量列表。
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """用于 YOLO 模型的区域注意力模块,提供高效的注意力机制。

    该模块实现了基于区域的注意力机制,以空间感知的方式处理输入特征,使其对目标检测任务特别有效。

    属性:
        area (int): 特征图被划分的区域数量。
        num_heads (int): 注意力机制被划分的头数。
        head_dim (int): 每个注意力头的维度。
        qkv (Conv): 用于计算查询、键和值张量的卷积层。
        proj (Conv): 投影卷积层。
        pe (Conv): 位置编码卷积层。

    方法:
        forward: 对输入张量应用区域注意力。

    示例:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """初始化用于 YOLO 模型的区域注意力模块。

        参数:
            dim (int): 隐藏通道数。
            num_heads (int): 注意力机制被划分的头数。
            area (int): 特征图被划分的区域数量。
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过区域注意力处理输入张量。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 经过区域注意力后的输出张量。
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """用于 YOLO 模型中高效特征提取的区域注意力块模块。

    该模块实现了区域注意力机制与前馈网络的结合,用于处理特征图。它使用一种新颖的基于区域的注意力方法,比传统的自注意力更高效,同时保持有效性。

    属性:
        attn (AAttn): 用于处理空间特征的区域注意力模块。
        mlp (nn.Sequential): 用于特征转换的多层感知器。

    方法:
        _init_weights: 使用截断正态分布初始化模块权重。
        forward: 对输入张量应用区域注意力和前馈处理。

    示例:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """初始化区域注意力块模块。

        参数:
            dim (int): 输入通道数。
            num_heads (int): 注意力机制被划分的头数。
            mlp_ratio (float): MLP 隐藏维度的扩展比率。
            area (int): 特征图被划分的区域数量。
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """使用截断正态分布初始化权重。

        参数:
            m (nn.Module): 要初始化的模块。
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行通过 ABlock 的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 经过区域注意力和前馈处理后的输出张量。
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """区域注意力 C2f 模块,使用基于区域的注意力机制增强特征提取。

    该模块通过结合区域注意力和 ABlock 层来扩展 C2f 架构,以改进特征处理。它支持区域注意力和标准卷积两种模式。

    属性:
        cv1 (Conv): 初始 1x1 卷积层,将输入通道减少到隐藏通道。
        cv2 (Conv): 最终 1x1 卷积层,处理连接的特征。
        gamma (nn.Parameter | None): 使用区域注意力时用于残差缩放的可学习参数。
        m (nn.ModuleList): 用于特征处理的 ABlock 或 C3k 模块列表。

    方法:
        forward: 通过区域注意力或标准卷积路径处理输入。

    示例:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """初始化区域注意力 C2f 模块。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 要堆叠的 ABlock 或 C3k 模块数量。
            a2 (bool): 是否使用区域注意力块。如果为 False,则使用 C3k 块。
            area (int): 特征图被划分的区域数量。
            residual (bool): 是否使用带有可学习 gamma 参数的残差连接。
            mlp_ratio (float): MLP 隐藏维度的扩展比率。
            e (float): 隐藏通道的通道扩展比率。
            g (int): 分组卷积的组数。
            shortcut (bool): 是否在 C3k 块中使用快捷连接。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行通过 A2C2f 层的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            (torch.Tensor): 处理后的输出张量。
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """用于基于 Transformer 架构的 SwiGLU 前馈网络。"""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """初始化 SwiGLU FFN,包含输入维度、输出维度和扩展因子。

        参数:
            gc (int): 引导通道数。
            ec (int): 嵌入通道数。
            e (int): 扩展因子。
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征应用 SwiGLU 转换。"""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """神经网络模块的残差连接包装器。"""

    def __init__(self, m: nn.Module) -> None:
        """初始化带有包装模块的残差模块。

        参数:
            m (nn.Module): 要用残差连接包装的模块。
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征应用残差连接。"""
        return x + self.m(x)


class SAVPE(nn.Module):
    """用于特征增强的空间感知视觉提示嵌入模块。"""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """初始化 SAVPE 模块,包含通道数、中间通道数和嵌入维度。

        参数:
            ch (list[int]): 输入通道维度列表。
            c3 (int): 中间通道数。
            embed (int): 嵌入维度。
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """处理输入特征和视觉提示以生成增强的嵌入。"""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
