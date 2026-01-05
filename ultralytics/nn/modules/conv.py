"""
卷积模块。

Convolution modules.

本模块包含YOLO系列模型中使用的各种卷积层和注意力机制，包括标准卷积、深度卷积、
Ghost卷积、RepConv等，以及CBAM、通道注意力、空间注意力等注意力模块。
"""

from __future__ import annotations  # 允许在类型注解中使用字符串形式的类型提示

import math  # 数学函数库

import numpy as np  # 数组和数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

__all__ = (
    "CBAM",  # 卷积注意力模块（Convolutional Block Attention Module）
    "ChannelAttention",  # 通道注意力模块
    "Concat",  # 张量拼接模块
    "Conv",  # 标准卷积模块
    "Conv2",  # 简化的RepConv模块
    "ConvTranspose",  # 转置卷积模块
    "DWConv",  # 深度可分离卷积模块
    "DWConvTranspose2d",  # 深度可分离转置卷积模块
    "Focus",  # Focus模块，用于特征聚焦
    "GhostConv",  # Ghost卷积模块
    "Index",  # 索引选择模块
    "LightConv",  # 轻量级卷积模块
    "RepConv",  # 重参数化卷积模块
    "SpatialAttention",  # 空间注意力模块
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    自动计算padding以实现'same'形状输出。

    Pad to 'same' shape outputs.

    Args:
        k (int | tuple): 卷积核大小
        p (int | tuple, optional): 填充大小。如果为None则自动计算
        d (int): 膨胀率

    Returns:
        (int | tuple): 计算得到的填充大小
    """
    if d > 1:
        # 计算实际的卷积核大小（考虑膨胀率）
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        # 自动计算padding以保持输出尺寸与输入相同
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    标准卷积模块，包含卷积、批归一化和激活函数。

    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): 卷积层
        bn (nn.BatchNorm2d): 批归一化层
        act (nn.Module): 激活函数层
        default_act (nn.Module): 默认激活函数（SiLU）
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化Conv层。

        Initialize Conv layer with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充大小
            g (int): 分组卷积的组数
            d (int): 膨胀率
            act (bool | nn.Module): 激活函数。True使用默认激活函数，False不使用激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        对输入张量应用卷积、批归一化和激活函数。

        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        应用卷积和激活函数（不使用批归一化）。

        Apply convolution and activation without batch normalization.

        用于推理时的融合模式，将BN层的参数融合到卷积层中以提高推理速度。

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    简化的RepConv模块，包含卷积融合。

    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): 主要的3x3卷积层
        cv2 (nn.Conv2d): 额外的1x1卷积层
        bn (nn.BatchNorm2d): 批归一化层
        act (nn.Module): 激活函数层
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        初始化Conv2层。

        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充大小
            g (int): 分组卷积的组数
            d (int): 膨胀率
            act (bool | nn.Module): 激活函数
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        对输入张量应用卷积、批归一化和激活函数。

        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        应用融合的卷积、批归一化和激活函数。

        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """
        融合并行卷积。

        Fuse parallel convolutions.

        将1x1卷积的权重融合到3x3卷积中，减少推理时的计算量。
        """
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    轻量级卷积模块，包含1x1卷积和深度卷积。

    Light convolution module with 1x1 and depthwise convolutions.

    本实现基于PaddleDetection的HGNetV2骨干网络。
    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1卷积层
        conv2 (DWConv): 深度卷积层
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        初始化LightConv层。

        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 深度卷积的卷积核大小
            act (nn.Module): 激活函数
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        对输入张量应用2个卷积。

        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """
    深度可分离卷积模块。

    Depth-wise convolution module.

    深度卷积通过为每个输入通道使用单独的卷积核来减少参数数量和计算量。
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        初始化深度卷积。

        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            d (int): 膨胀率
            act (bool | nn.Module): 激活函数
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """
    深度可分离转置卷积模块。

    Depth-wise transpose convolution module.

    用于上采样操作。
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        初始化深度可分离转置卷积。

        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p1 (int): 填充
            p2 (int): 输出填充
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    转置卷积模块，可选批归一化和激活函数。

    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): 转置卷积层
        bn (nn.BatchNorm2d | nn.Identity): 批归一化层
        act (nn.Module): 激活函数层
        default_act (nn.Module): 默认激活函数（SiLU）
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        初始化ConvTranspose层。

        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int): 填充
            bn (bool): 是否使用批归一化
            act (bool | nn.Module): 激活函数
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        对输入应用转置卷积、批归一化和激活函数。

        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        对输入应用激活函数和转置卷积操作。

        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus模块，用于集中特征信息。

    Focus module for concentrating feature information.

    将输入张量切片为4部分并在通道维度上拼接。
    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): 卷积层
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        初始化Focus模块。

        Initialize Focus module with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            g (int): 分组卷积的组数
            act (bool | nn.Module): 激活函数
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        对输入张量应用Focus操作和卷积。

        Apply Focus operation and convolution to input tensor.

        输入形状为(B, C, W, H)，输出形状为(B, 4C, W/2, H/2)。
        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost卷积模块。

    Ghost Convolution module.

    通过使用廉价操作生成更多特征，从而用更少的参数生成更多的特征。
    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): 主卷积
        cv2 (Conv): 廉价操作卷积

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        初始化Ghost卷积模块。

        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            g (int): 分组卷积的组数
            act (bool | nn.Module): 激活函数
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        对输入张量应用Ghost卷积。

        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 拼接特征的输出张量
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv模块，具有训练和部署模式。

    RepConv module with training and deploy modes.

    该模块用于RT-DETR，可以在推理时融合卷积以提高效率。
    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3卷积
        conv2 (Conv): 1x1卷积
        bn (nn.BatchNorm2d, optional): 恒等分支的批归一化
        act (nn.Module): 激活函数
        default_act (nn.Module): 默认激活函数（SiLU）

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        初始化RepConv模块。

        Initialize RepConv module with given parameters.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int): 填充
            g (int): 分组卷积的组数
            d (int): 膨胀率
            act (bool | nn.Module): 激活函数
            bn (bool): 是否对恒等分支使用批归一化
            deploy (bool): 推理模式的部署标志
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        部署模式的前向传播。

        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        训练模式的前向传播。

        Forward pass for training mode.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        通过融合卷积计算等效的卷积核和偏置。

        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): 等效卷积核
            (torch.Tensor): 等效偏置
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        将1x1卷积核填充到3x3大小。

        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1卷积核

        Returns:
            (torch.Tensor): 填充后的3x3卷积核
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        将批归一化与卷积权重融合。

        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): 要融合的分支

        Returns:
            kernel (torch.Tensor): 融合后的卷积核
            bias (torch.Tensor): 融合后的偏置
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """
        融合卷积以进行推理，创建单个等效卷积。

        Fuse convolutions for inference by creating a single equivalent convolution.
        """
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    通道注意力模块，用于特征重标定。

    Channel-attention module for feature recalibration.

    基于全局平均池化对通道应用注意力权重。
    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): 全局平均池化
        fc (nn.Conv2d): 全连接层，实现为1x1卷积
        act (nn.Sigmoid): 用于注意力权重的Sigmoid激活

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        初始化通道注意力模块。

        Initialize Channel-attention module.

        Args:
            channels (int): 输入通道数
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用通道注意力。

        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 通道注意力后的输出张量
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    空间注意力模块，用于特征重标定。

    Spatial-attention module for feature recalibration.

    基于通道统计信息对空间维度应用注意力权重。
    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): 空间注意力的卷积层
        act (nn.Sigmoid): 用于注意力权重的Sigmoid激活
    """

    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块。

        Initialize Spatial-attention module.

        Args:
            kernel_size (int): 卷积核大小（3或7）
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        对输入张量应用空间注意力。

        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 空间注意力后的输出张量
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    卷积块注意力模块。

    Convolutional Block Attention Module.

    结合通道和空间注意力机制以实现全面的特征细化。
    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): 通道注意力模块
        spatial_attention (SpatialAttention): 空间注意力模块
    """

    def __init__(self, c1, kernel_size=7):
        """
        初始化CBAM。

        Initialize CBAM with given parameters.

        Args:
            c1 (int): 输入通道数
            kernel_size (int): 空间注意力的卷积核大小
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        依次对输入张量应用通道和空间注意力。

        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 注意力后的输出张量
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    沿指定维度拼接张量列表。

    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): 沿其拼接张量的维度
    """

    def __init__(self, dimension=1):
        """
        初始化Concat模块。

        Initialize Concat module.

        Args:
            dimension (int): 沿其拼接张量的维度
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        沿指定维度拼接输入张量。

        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): 输入张量列表

        Returns:
            (torch.Tensor): 拼接后的张量
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    返回输入的特定索引。

    Returns a particular index of the input.

    Attributes:
        index (int): 从输入中选择的索引
    """

    def __init__(self, index=0):
        """
        初始化Index模块。

        Initialize Index module.

        Args:
            index (int): 从输入中选择的索引
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        从输入中选择并返回特定索引。

        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): 输入张量列表

        Returns:
            (torch.Tensor): 选择的张量
        """
        return x[self.index]
