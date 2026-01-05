"""
神经网络模块工具函数

本模块包含了神经网络构建和训练中常用的工具函数，包括:
- _get_clones: 克隆模块
- bias_init_with_prob: 基于概率初始化偏置
- linear_init: 线性层初始化
- inverse_sigmoid: 逆sigmoid函数
- multi_scale_deformable_attn_pytorch: 多尺度可变形注意力的PyTorch实现
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"


def _get_clones(module, n):
    """
    创建模块的克隆列表

    该函数通过深拷贝的方式创建多个相同的模块副本。
    常用于构建多层Transformer等需要重复相同结构的网络。

    Args:
        module (nn.Module): 要克隆的模块
        n (int): 克隆数量

    Returns:
        (nn.ModuleList): 包含n个模块克隆的ModuleList

    示例:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    # 使用深拷贝确保每个模块都是独立的，拥有独立的参数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """
    根据先验概率初始化卷积/全连接层的偏置值

    该函数基于先验概率计算偏置初始化值。在目标检测模型中，通常用于初始化分类层，
    使得网络在训练初期就有合理的正样本预测概率，从而加速训练收敛。

    公式推导：
    - 假设我们希望初始预测概率为 p
    - sigmoid(bias) = p
    - bias = log(p / (1 - p))
    - 为了稳定训练，通常使用负值: -log((1-p) / p)

    Args:
        prior_prob (float, optional): 先验概率，默认0.01（即1%的正样本概率）

    Returns:
        (float): 计算得到的偏置初始化值

    示例:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"偏置初始化值: {bias:.4f}")
        偏置初始化值: -4.5951
    """
    # 计算逆sigmoid: -log((1-p)/p)
    return float(-np.log((1 - prior_prob) / prior_prob))


def linear_init(module):
    """
    初始化线性层的权重和偏置

    该函数使用均匀分布初始化线性层的权重和偏置。
    边界范围根据输入维度计算，确保初始化值的方差合理。

    初始化方法:
    - 计算边界: bound = 1 / sqrt(fan_in)
    - 权重从 [-bound, bound] 均匀分布采样
    - 偏置（如果存在）也从 [-bound, bound] 均匀分布采样

    这种初始化方法有助于避免梯度消失或爆炸问题。

    Args:
        module (nn.Module): 要初始化的线性模块

    Returns:
        (nn.Module): 初始化后的模块

    示例:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> initialized_linear = linear_init(linear)
    """
    # 根据输入维度计算初始化边界
    bound = 1 / math.sqrt(module.weight.shape[0])
    # 权重使用均匀分布初始化
    uniform_(module.weight, -bound, bound)
    # 如果有偏置，也使用均匀分布初始化
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """
    计算张量的逆sigmoid函数

    逆sigmoid函数（也称为logit函数）是sigmoid函数的反函数。
    在神经网络中，特别是在注意力机制和坐标变换中非常有用。

    数学定义:
    - sigmoid(x) = 1 / (1 + exp(-x))
    - inverse_sigmoid(p) = log(p / (1 - p))

    应用场景:
    - RT-DETR中将归一化坐标[0,1]转换到logit空间
    - 可变形注意力中的参考点变换
    - 需要将概率值转换回logits的场景

    Args:
        x (torch.Tensor): 输入张量，值应在[0, 1]范围内
        eps (float, optional): 小的epsilon值，防止数值不稳定（除零或log(0)）

    Returns:
        (torch.Tensor): 应用逆sigmoid函数后的张量

    示例:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    # 将输入限制在[0, 1]范围内
    x = x.clamp(min=0, max=1)
    # 添加eps避免log(0)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    # 计算 log(x / (1-x))
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    多尺度可变形注意力的PyTorch实现

    该函数实现了跨多个特征图尺度的可变形注意力机制，允许模型通过学习的偏移量
    关注不同的空间位置。这是Deformable DETR的核心组件。

    工作流程:
    1. 将多尺度value特征按空间尺寸分割
    2. 对每个尺度，根据采样位置使用grid_sample进行双线性插值采样
    3. 使用注意力权重对采样值进行加权聚合
    4. 返回最终的注意力输出

    Args:
        value (torch.Tensor): Value张量，shape: (bs, num_keys, num_heads, embed_dims)
                             其中num_keys = sum(H_i * W_i)，所有层级的空间位置总和
        value_spatial_shapes (torch.Tensor): 各层级特征图的空间形状，shape: (num_levels, 2)
                                            每行为[H, W]
        sampling_locations (torch.Tensor): 采样位置，shape: (bs, num_queries, num_heads, num_levels,
                                          num_points, 2)
                                          值在[0,1]范围内，表示归一化的空间坐标
        attention_weights (torch.Tensor): 注意力权重，shape: (bs, num_queries, num_heads, num_levels,
                                         num_points)

    Returns:
        (torch.Tensor): 输出张量，shape: (bs, num_queries, embed_dims)

    参考:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    # 1. 按空间尺寸分割value：将多尺度特征分割为列表
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # 2. 将采样位置从[0,1]转换到grid_sample所需的[-1,1]范围
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    # 3. 对每个尺度进行采样
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # 重塑value以适应grid_sample的输入格式
        # (bs, H_*W_, num_heads, embed_dims) -> (bs*num_heads, embed_dims, H_, W_)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)

        # 提取当前层级的采样网格
        # (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # 使用双线性插值在采样位置提取特征
        # 输出shape: (bs*num_heads, embed_dims, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # 4. 重塑注意力权重用于加权聚合
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    # 5. 加权聚合：将所有层级的采样值按注意力权重加权求和
    # stack: (num_levels, bs*num_heads, embed_dims, num_queries, num_points)
    # flatten(-2): (bs*num_heads, embed_dims, num_queries, num_levels*num_points)
    # 乘以注意力权重并沿最后一维求和
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )

    # 6. 转换维度顺序: (bs, embed_dims, num_queries) -> (bs, num_queries, embed_dims)
    return output.transpose(1, 2).contiguous()
