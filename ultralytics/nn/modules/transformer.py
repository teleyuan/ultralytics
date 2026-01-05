"""
Transformer模块

本模块包含了各种Transformer架构的实现，用于YOLO系列模型，包括:
- TransformerEncoderLayer: 标准Transformer编码器层
- AIFI: 用于2D数据的注意力增强特征交互层
- TransformerLayer: 简化的Transformer层
- TransformerBlock: 完整的Transformer块
- MLPBlock/MLP: 多层感知机模块
- LayerNorm2d: 2D层归一化
- MSDeformAttn: 多尺度可变形注意力
- DeformableTransformerDecoderLayer: 可变形Transformer解码器层
- DeformableTransformerDecoder: 可变形Transformer解码器
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器的单层实现

    该类实现了标准的Transformer编码器层，包含多头注意力和前馈网络，
    支持预归一化(Pre-LN)和后归一化(Post-LN)两种配置。

    属性:
        ma (nn.MultiheadAttention): 多头注意力模块
        fc1 (nn.Linear): 前馈网络的第一个线性层
        fc2 (nn.Linear): 前馈网络的第二个线性层
        norm1 (nn.LayerNorm): 注意力后的层归一化
        norm2 (nn.LayerNorm): 前馈网络后的层归一化
        dropout (nn.Dropout): 前馈网络的Dropout层
        dropout1 (nn.Dropout): 注意力后的Dropout层
        dropout2 (nn.Dropout): 前馈网络后的Dropout层
        act (nn.Module): 激活函数（默认GELU）
        normalize_before (bool): 是否在注意力和前馈网络之前应用归一化
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        初始化TransformerEncoderLayer

        Args:
            c1 (int): 输入维度（嵌入维度）
            cm (int): 前馈网络的隐藏维度，通常为嵌入维度的4倍
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
            act (nn.Module): 激活函数
            normalize_before (bool): 是否使用Pre-LN（True）还是Post-LN（False）
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        # 检查PyTorch版本，需要>=1.9才能使用batch_first参数
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() 需要 torch>=1.9 才能使用 nn.MultiheadAttention(batch_first=True)。"
            )

        # 多头注意力层
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络实现：两个线性层 + 激活函数
        self.fc1 = nn.Linear(c1, cm)  # 扩展维度
        self.fc2 = nn.Linear(cm, c1)  # 恢复维度

        # 层归一化
        self.norm1 = nn.LayerNorm(c1)  # 注意力后的归一化
        self.norm2 = nn.LayerNorm(c1)  # 前馈网络后的归一化

        # Dropout层
        self.dropout = nn.Dropout(dropout)  # 前馈网络中的Dropout
        self.dropout1 = nn.Dropout(dropout)  # 注意力后的Dropout
        self.dropout2 = nn.Dropout(dropout)  # 前馈网络后的Dropout

        self.act = act  # 激活函数
        self.normalize_before = normalize_before  # 归一化顺序标志

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        将位置编码添加到张量

        Args:
            tensor: 输入张量
            pos: 位置编码张量（可选）

        Returns:
            添加位置编码后的张量，如果pos为None则返回原张量
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        使用后归一化(Post-LN)的前向传播

        顺序: 注意力 -> Add&Norm -> FFN -> Add&Norm

        Args:
            src (torch.Tensor): 输入张量
            src_mask (torch.Tensor, optional): 序列掩码，用于屏蔽某些位置
            src_key_padding_mask (torch.Tensor, optional): 批次填充掩码
            pos (torch.Tensor, optional): 位置编码

        Returns:
            (torch.Tensor): 经过注意力和前馈网络处理后的输出张量
        """
        # 1. 多头自注意力
        q = k = self.with_pos_embed(src, pos)  # 为query和key添加位置编码
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 归一化

        # 2. 前馈网络 (FFN)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))  # fc1 -> act -> dropout -> fc2
        src = src + self.dropout2(src2)  # 残差连接
        return self.norm2(src)  # 归一化

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        使用预归一化(Pre-LN)的前向传播

        顺序: Norm -> 注意力 -> Add -> Norm -> FFN -> Add
        Pre-LN通常训练更稳定，是现代Transformer的主流选择

        Args:
            src (torch.Tensor): 输入张量
            src_mask (torch.Tensor, optional): 序列掩码
            src_key_padding_mask (torch.Tensor, optional): 批次填充掩码
            pos (torch.Tensor, optional): 位置编码

        Returns:
            (torch.Tensor): 经过注意力和前馈网络处理后的输出张量
        """
        # 1. 多头自注意力（先归一化）
        src2 = self.norm1(src)  # 归一化
        q = k = self.with_pos_embed(src2, pos)  # 为query和key添加位置编码
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接

        # 2. 前馈网络（先归一化）
        src2 = self.norm2(src)  # 归一化
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))  # FFN
        return src + self.dropout2(src2)  # 残差连接

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        编码器层的前向传播

        根据normalize_before标志选择使用Pre-LN或Post-LN

        Args:
            src (torch.Tensor): 输入张量
            src_mask (torch.Tensor, optional): 序列掩码
            src_key_padding_mask (torch.Tensor, optional): 批次填充掩码
            pos (torch.Tensor, optional): 位置编码

        Returns:
            (torch.Tensor): Transformer编码器层处理后的输出张量
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """
    AIFI (Attention-based Feature Interaction) Transformer层

    该类扩展了TransformerEncoderLayer，专门用于处理2D特征图。
    通过添加2D正余弦位置编码并适当处理空间维度来实现。

    AIFI用于增强特征之间的交互，常用于YOLO模型的颈部网络。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        初始化AIFI实例

        Args:
            c1 (int): 输入维度（通道数）
            cm (int): 前馈网络的隐藏维度
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
            act (nn.Module): 激活函数
            normalize_before (bool): 是否在注意力和前馈网络之前应用归一化
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AIFI Transformer层的前向传播

        处理流程:
        1. 生成2D位置编码
        2. 将空间维度展平: [B,C,H,W] -> [B,H*W,C]
        3. 应用Transformer编码器层
        4. 恢复空间维度: [B,H*W,C] -> [B,C,H,W]

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            (torch.Tensor): 输出张量，形状为 [B, C, H, W]
        """
        c, h, w = x.shape[1:]  # 获取通道数和空间维度

        # 构建2D正余弦位置编码
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)

        # 展平空间维度并应用Transformer
        # [B, C, H, W] -> [B, H*W, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))

        # 恢复空间维度
        # [B, H*W, C] -> [B, C, H, W]
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """
        构建2D正余弦位置编码

        使用正弦和余弦函数为2D空间位置生成位置编码。
        与原始Transformer中的1D位置编码类似，但扩展到2D空间。

        Args:
            w (int): 特征图宽度
            h (int): 特征图高度
            embed_dim (int): 嵌入维度，必须能被4整除
            temperature (float): 温度参数，用于控制正余弦函数的频率

        Returns:
            (torch.Tensor): 位置编码，形状为 [1, embed_dim, h*w]
                           包含 [sin(x), cos(x), sin(y), cos(y)] 的拼接
        """
        # 确保嵌入维度能被4整除（x和y各需要sin和cos）
        assert embed_dim % 4 == 0, "嵌入维度必须能被4整除才能使用2D正余弦位置编码"

        # 创建空间网格
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)

        # 计算频率参数
        pos_dim = embed_dim // 4  # 每个维度(x,y)各占一半，每一半又分为sin和cos
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)  # 频率衰减

        # 计算位置编码
        out_w = grid_w.flatten()[..., None] @ omega[None]  # (h*w, pos_dim)
        out_h = grid_h.flatten()[..., None] @ omega[None]  # (h*w, pos_dim)

        # 拼接sin和cos编码: [sin(x), cos(x), sin(y), cos(y)]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """
    简化的Transformer层

    基于 https://arxiv.org/abs/2010.11929
    为了更好的性能，移除了LayerNorm层，只保留核心的自注意力和前馈网络。

    结构:
    1. Q, K, V线性投影
    2. 多头自注意力
    3. 两层前馈网络
    每个子模块都使用残差连接
    """

    def __init__(self, c: int, num_heads: int):
        """
        初始化简化的Transformer层

        Args:
            c (int): 输入和输出通道维度
            num_heads (int): 注意力头数
        """
        super().__init__()
        # Q, K, V投影层
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # 多头自注意力
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # 前馈网络的两层全连接
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer层的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        # 自注意力 + 残差连接
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        # 前馈网络 + 残差连接
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """
    视觉Transformer块

    基于 https://arxiv.org/abs/2010.11929
    该类实现了一个完整的Transformer块，用于处理2D特征图。

    主要组件:
    - 可选的卷积层：用于通道调整
    - 可学习的位置编码：使用线性层生成位置编码
    - 多层Transformer层：堆叠的自注意力和前馈网络

    处理流程:
    1. 将2D特征图 [b, c, h, w] 展平为序列 [h*w, b, c]
    2. 添加可学习的位置编码
    3. 通过多层Transformer处理
    4. 重塑回2D特征图格式

    属性:
        conv (Conv, optional): 如果输入输出通道数不同时的卷积层
        linear (nn.Linear): 可学习的位置编码生成器
        tr (nn.Sequential): Transformer层的序列容器
        c2 (int): 输出通道维度
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """
        初始化Transformer块

        Args:
            c1 (int): 输入通道维度
            c2 (int): 输出通道维度
            num_heads (int): 注意力头数
            num_layers (int): Transformer层数
        """
        super().__init__()
        self.conv = None
        # 如果输入输出通道数不同，使用卷积调整
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # 可学习的位置编码
        self.linear = nn.Linear(c2, c2)
        # 堆叠多个Transformer层
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer块的前向传播

        Args:
            x (torch.Tensor): 输入张量，shape: [b, c1, h, w]

        Returns:
            (torch.Tensor): 输出张量，shape: [b, c2, h, w]
        """
        # 通道调整（如果需要）
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        # 将2D特征图展平为序列: [b, c, h, w] -> [h*w, b, c]
        p = x.flatten(2).permute(2, 0, 1)
        # 添加位置编码并通过Transformer层，最后重塑为2D
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(nn.Module):
    """
    多层感知机块（单个块）

    实现了一个简单的两层MLP：
    Linear -> Activation -> Linear

    这是Vision Transformer中FFN（前馈网络）的标准实现。
    """

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """
        初始化MLPBlock

        Args:
            embedding_dim (int): 输入和输出维度
            mlp_dim (int): 隐藏层维度（通常是embedding_dim的4倍）
            act (nn.Module): 激活函数，默认GELU
        """
        super().__init__()
        # 第一层：升维
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        # 第二层：降维
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLPBlock的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        # Linear -> Activation -> Linear
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron，也称为前馈网络FFN）

    实现了一个可配置的MLP，支持多层线性层、激活函数和可选的输出处理。

    特性:
    - 可配置的层数和维度
    - 可选的残差连接
    - 可选的输出归一化
    - 可选的sigmoid输出激活

    属性:
        num_layers (int): MLP的层数
        layers (nn.ModuleList): 线性层列表
        sigmoid (bool): 是否对输出应用sigmoid
        act (nn.Module): 激活函数
        residual (bool): 是否使用残差连接
        out_norm (nn.Module): 输出归一化层
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: nn.Module = None,
    ):
        """
        初始化MLP

        Args:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            num_layers (int): 层数
            act (nn.Module): 激活函数，默认ReLU
            sigmoid (bool): 是否对输出应用sigmoid
            residual (bool): 是否使用残差连接（仅当input_dim == output_dim时有效）
            out_norm (nn.Module, optional): 输出归一化层
        """
        super().__init__()
        self.num_layers = num_layers
        # 构建各层的维度列表: [input_dim, hidden_dim, ..., hidden_dim, output_dim]
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()
        # 残差连接只在输入输出维度相同时支持
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # 输出归一化层（可选）
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): 输出张量
        """
        orig_x = x
        # 逐层前向传播，最后一层不使用激活函数
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 残差连接
        if getattr(self, "residual", False):
            x = x + orig_x
        # 输出归一化
        x = getattr(self, "out_norm", nn.Identity())(x)
        # 可选的sigmoid激活
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """
    2D层归一化模块

    受Detectron2和ConvNeXt实现的启发。该类实现了用于2D特征图的层归一化，
    在通道维度上进行归一化，同时保留空间维度。

    与BatchNorm的区别:
    - BatchNorm: 在batch和空间维度上归一化，通道独立
    - LayerNorm: 在通道维度上归一化，batch和空间位置独立

    对于输入 [b, c, h, w]：
    - 计算每个位置(b, h, w)上所有通道的均值和方差
    - 对该位置的所有通道进行归一化

    属性:
        weight (nn.Parameter): 可学习的缩放参数，shape: [num_channels]
        bias (nn.Parameter): 可学习的偏移参数，shape: [num_channels]
        eps (float): 数值稳定性常数

    参考:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        初始化LayerNorm2d

        Args:
            num_channels (int): 输入的通道数
            eps (float): 数值稳定性常数，默认1e-6
        """
        super().__init__()
        # 可学习的仿射变换参数
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D层归一化的前向传播

        Args:
            x (torch.Tensor): 输入张量，shape: [b, c, h, w]

        Returns:
            (torch.Tensor): 归一化后的输出张量，shape: [b, c, h, w]
        """
        # 计算通道维度的均值: [b, 1, h, w]
        u = x.mean(1, keepdim=True)
        # 计算通道维度的方差: [b, 1, h, w]
        s = (x - u).pow(2).mean(1, keepdim=True)
        # 归一化: (x - mean) / sqrt(var + eps)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 仿射变换: weight * x + bias
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力模块 (Multiscale Deformable Attention)

    基于Deformable-DETR和PaddleDetection的实现。
    该模块实现了多尺度可变形注意力，能够通过可学习的采样位置和注意力权重
    关注多个尺度的特征。

    核心思想:
    - 传统注意力需要计算所有位置的注意力，复杂度为O(N²)
    - 可变形注意力只在少数几个采样点计算注意力，复杂度为O(N)
    - 采样位置是可学习的，能够自适应地关注重要区域

    属性:
        im2col_step (int): im2col操作的步长
        d_model (int): 模型维度
        n_levels (int): 特征层级数量
        n_heads (int): 注意力头数
        n_points (int): 每个注意力头在每个特征层级的采样点数
        sampling_offsets (nn.Linear): 生成采样偏移的线性层
        attention_weights (nn.Linear): 生成注意力权重的线性层
        value_proj (nn.Linear): 投影value的线性层
        output_proj (nn.Linear): 投影输出的线性层

    参考:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """
        初始化多尺度可变形注意力模块

        Args:
            d_model (int): 模型维度（嵌入维度）
            n_levels (int): 特征层级数量（多尺度）
            n_heads (int): 注意力头数
            n_points (int): 每个注意力头在每个特征层级的采样点数
        """
        super().__init__()
        # 确保模型维度能被头数整除
        if d_model % n_heads != 0:
            raise ValueError(f"d_model必须能被n_heads整除，但得到{d_model}和{n_heads}")
        _d_per_head = d_model // n_heads
        # 注意：将_d_per_head设为2的幂次在CUDA实现中更高效
        assert _d_per_head * n_heads == d_model, "`d_model`必须能被`n_heads`整除"

        self.im2col_step = 64  # im2col操作的步长

        # 保存配置参数
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 采样偏移生成层：为每个头、每个层级、每个采样点生成2D偏移(x,y)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # 注意力权重生成层：为每个头、每个层级、每个采样点生成权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # Value投影层
        self.value_proj = nn.Linear(d_model, d_model)

        # 输出投影层
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()  # 初始化参数

    def _reset_parameters(self):
        """
        重置模块参数

        采样偏移的初始化非常重要：
        - 将偏移初始化为围绕参考点的环形分布
        - 每个注意力头关注不同的方向
        - 不同的采样点有不同的距离
        """
        # 采样偏移权重初始化为0
        constant_(self.sampling_offsets.weight.data, 0.0)

        # 采样偏移偏置初始化为环形分布
        # 每个头看向不同的方向（均匀分布在圆周上）
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # (n_heads, 2)
        # 归一化到单位圆
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        # 不同的采样点有不同的距离（1, 2, 3, ...）
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # 注意力权重初始化
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

        # Value和输出投影层使用Xavier初始化
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        多尺度可变形注意力的前向传播

        核心步骤:
        1. 投影value
        2. 从query生成采样偏移和注意力权重
        3. 根据参考框和偏移计算实际采样位置
        4. 在多个尺度的特征图上采样
        5. 加权聚合采样特征

        Args:
            query (torch.Tensor): 查询张量，形状 [bs, query_length, C]
            refer_bbox (torch.Tensor): 参考边界框，形状 [bs, query_length, n_levels, 2]
                                      范围在[0,1]，左上角(0,0)，右下角(1,1)
            value (torch.Tensor): Value张量，形状 [bs, value_length, C]
            value_shapes (list): 特征图形状列表 [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (torch.Tensor, optional): 掩码张量，形状 [bs, value_length]
                                                True表示非填充元素，False表示填充元素

        Returns:
            (torch.Tensor): 输出张量，形状 [bs, Length_{query}, C]

        参考:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]  # 批次大小和查询长度
        len_v = value.shape[1]  # value长度
        # 确保value长度等于所有层级的特征图大小之和
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        # 1. 投影value
        value = self.value_proj(value)
        if value_mask is not None:
            # 将填充位置的value设为0
            value = value.masked_fill(value_mask[..., None], float(0))
        # 重塑为多头形式
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        # 2. 生成采样偏移和注意力权重
        # 采样偏移：(bs, len_q, n_heads, n_levels, n_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)

        # 注意力权重：(bs, len_q, n_heads, n_levels, n_points)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)

        # 3. 计算采样位置
        # 根据参考框格式计算实际采样位置
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            # 参考点格式：(x, y) 中心点
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            # 参考点格式：(x, y, w, h) 边界框
            # 偏移相对于边界框大小
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"reference_points的最后一维必须是2或4，但得到{num_points}。")

        # 4. 执行多尺度可变形注意力采样
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)

        # 5. 输出投影
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    可变形Transformer解码器层

    基于PaddleDetection和Deformable-DETR的实现。
    该类实现了一个完整的解码器层，包含:
    1. 自注意力 (Self-Attention)
    2. 交叉注意力 (Cross-Attention) - 使用多尺度可变形注意力
    3. 前馈网络 (FFN)

    属性:
        self_attn (nn.MultiheadAttention): 自注意力模块
        dropout1 (nn.Dropout): 自注意力后的Dropout
        norm1 (nn.LayerNorm): 自注意力后的层归一化
        cross_attn (MSDeformAttn): 交叉注意力模块（多尺度可变形）
        dropout2 (nn.Dropout): 交叉注意力后的Dropout
        norm2 (nn.LayerNorm): 交叉注意力后的层归一化
        linear1 (nn.Linear): 前馈网络的第一个线性层
        act (nn.Module): 激活函数
        dropout3 (nn.Dropout): 前馈网络中的Dropout
        linear2 (nn.Linear): 前馈网络的第二个线性层
        dropout4 (nn.Dropout): 前馈网络后的Dropout
        norm3 (nn.LayerNorm): 前馈网络后的层归一化

    参考:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """
        初始化可变形Transformer解码器层

        Args:
            d_model (int): 模型维度
            n_heads (int): 注意力头数
            d_ffn (int): 前馈网络维度
            dropout (float): Dropout概率
            act (nn.Module): 激活函数
            n_levels (int): 特征层级数量
            n_points (int): 采样点数量
        """
        super().__init__()

        # 自注意力模块：query与自身之间的注意力，用于query之间的信息交互
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力模块：使用多尺度可变形注意力，query关注编码器的特征图
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络模块：两层全连接网络，用于特征变换
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """
        将位置编码添加到输入张量

        Args:
            tensor (torch.Tensor): 输入张量
            pos (torch.Tensor | None): 位置编码，如果为None则不添加

        Returns:
            (torch.Tensor): 添加位置编码后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        前馈网络的前向传播

        Args:
            tgt (torch.Tensor): 输入张量

        Returns:
            (torch.Tensor): FFN输出张量
        """
        # 两层全连接网络: Linear -> Activation -> Dropout -> Linear
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        # 残差连接
        tgt = tgt + self.dropout4(tgt2)
        # 层归一化
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        可变形Transformer解码器层的前向传播

        执行顺序: 自注意力 -> 交叉注意力 -> 前馈网络
        每个模块都使用残差连接和层归一化

        Args:
            embed (torch.Tensor): 输入的query embedding
            refer_bbox (torch.Tensor): 参考边界框，用于可变形注意力的采样位置
            feats (torch.Tensor): 编码器的特征图
            shapes (list): 各层特征图的空间尺寸
            padding_mask (torch.Tensor, optional): 填充掩码
            attn_mask (torch.Tensor, optional): 注意力掩码
            query_pos (torch.Tensor, optional): Query位置编码

        Returns:
            (torch.Tensor): 解码器层输出
        """
        # 1. 自注意力：query之间的信息交互
        # 将query embedding与位置编码相加作为Q和K
        q = k = self.with_pos_embed(embed, query_pos)
        # 执行多头自注意力，V使用原始embed（不加位置编码）
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        # 残差连接 + Dropout + 层归一化
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # 2. 交叉注意力：query关注编码器特征
        # 使用多尺度可变形注意力，参考边界框用于确定采样位置
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        # 残差连接 + Dropout + 层归一化
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # 3. 前馈网络：特征变换
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    可变形Transformer解码器

    基于PaddleDetection的实现。该类实现了一个完整的可变形Transformer解码器，
    包含多个解码器层和用于边界框回归和分类的预测头。

    解码器采用迭代优化的方式：
    - 每个解码器层都会预测边界框和类别分数
    - 后续层使用前一层的预测结果作为参考边界框
    - 通过多层迭代逐步优化检测结果

    属性:
        layers (nn.ModuleList): 解码器层列表
        num_layers (int): 解码器层数
        hidden_dim (int): 隐藏维度
        eval_idx (int): 评估时使用的层索引，-1表示最后一层

    参考:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """
        初始化可变形Transformer解码器

        Args:
            hidden_dim (int): 隐藏维度
            decoder_layer (nn.Module): 解码器层模块
            num_layers (int): 解码器层数
            eval_idx (int): 评估时使用的层索引，默认-1表示最后一层
        """
        super().__init__()
        # 复制多个解码器层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # 处理负索引，例如-1表示最后一层
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # query embeddings
        refer_bbox: torch.Tensor,  # 初始参考边界框（锚点）
        feats: torch.Tensor,  # 编码器输出的图像特征
        shapes: list,  # 各层特征图的形状
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        解码器的前向传播

        迭代优化过程：
        1. 初始化参考边界框（通常来自top-k选择）
        2. 通过每个解码器层处理query embedding
        3. 每层预测边界框偏移和类别分数
        4. 将预测偏移应用到参考边界框，得到优化后的边界框
        5. 优化后的边界框作为下一层的参考边界框

        Args:
            embed (torch.Tensor): Query embeddings，shape: [bs, num_queries, hidden_dim]
            refer_bbox (torch.Tensor): 初始参考边界框，shape: [bs, num_queries, 4]
            feats (torch.Tensor): 编码器输出的图像特征
            shapes (list): 各层特征图的空间尺寸
            bbox_head (nn.Module): 边界框预测头（每层一个）
            score_head (nn.Module): 类别分数预测头（每层一个）
            pos_mlp (nn.Module): 位置编码MLP，将边界框转换为位置编码
            attn_mask (torch.Tensor, optional): 注意力掩码
            padding_mask (torch.Tensor, optional): 填充掩码

        Returns:
            dec_bboxes (torch.Tensor): 各层预测的边界框，shape: [num_layers, bs, num_queries, 4]
            dec_cls (torch.Tensor): 各层预测的类别分数，shape: [num_layers, bs, num_queries, num_classes]
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        # 将参考边界框归一化到[0,1]范围
        refer_bbox = refer_bbox.sigmoid()

        # 逐层迭代优化
        for i, layer in enumerate(self.layers):
            # 1. 通过解码器层处理
            # pos_mlp将边界框坐标转换为位置编码
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            # 2. 预测边界框偏移
            bbox = bbox_head[i](output)
            # 将偏移应用到参考边界框上，得到优化后的边界框
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            # 3. 保存预测结果
            if self.training:
                # 训练时保存所有层的预测，用于深度监督
                dec_cls.append(score_head[i](output))
                if i == 0:
                    # 第一层直接使用优化后的边界框
                    dec_bboxes.append(refined_bbox)
                else:
                    # 后续层计算相对于上一层预测的偏移
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                # 推理时只使用指定层的预测
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            # 4. 更新参考边界框
            last_refined_bbox = refined_bbox
            # 训练时detach防止梯度回传，推理时不需要
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
