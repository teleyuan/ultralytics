"""
Ultralytics 神经网络模块

这个模块提供了 Ultralytics 模型中使用的各种神经网络组件的访问接口，包括卷积块、
注意力机制、Transformer 组件以及检测/分割头部。

主要组件分类:
    - 卷积模块 (conv.py): 基础卷积层、注意力卷积、深度可分离卷积等
    - 块模块 (block.py): 瓶颈块、CSP 块、注意力块等复合模块
    - 头部模块 (head.py): 检测头、分割头、分类头、姿态估计头等
    - Transformer 模块 (transformer.py): 自注意力、可变形注意力、MLP 等

使用示例:
    使用 Netron 可视化模块结构
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)  # 创建输入张量
    >>> m = Conv(128, 128)  # 实例化卷积模块
    >>> f = f"{m._get_name()}.onnx"  # 定义导出文件名
    >>> torch.onnx.export(m, x, f)  # 导出为 ONNX 格式
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # 优化并可视化
"""

# 从 block 模块导入各种块结构
# 包括 CSP 块、瓶颈块、注意力块、池化块等复合模块
from .block import (
    C1,  # CSP 单卷积瓶颈块
    C2,  # CSP 双卷积瓶颈块
    C2PSA,  # C2 + 位置敏感注意力块
    C3,  # CSP 三卷积瓶颈块
    C3TR,  # C3 + Transformer 块
    CIB,  # 紧凑倒置块
    DFL,  # 分布式焦点损失积分模块
    ELAN1,  # ELAN 高效层聚合网络块
    PSA,  # 位置敏感注意力模块
    SPP,  # 空间金字塔池化层
    SPPELAN,  # SPP-ELAN 块
    SPPF,  # 快速空间金字塔池化层
    A2C2f,  # 区域注意力 C2f 模块
    AConv,  # 自适应卷积下采样
    ADown,  # 自适应下采样模块
    Attention,  # 多头自注意力模块
    BNContrastiveHead,  # 批归一化对比学习头
    Bottleneck,  # 标准瓶颈块
    BottleneckCSP,  # CSP 瓶颈块
    C2f,  # CSP 快速实现（2卷积）
    C2fAttn,  # C2f + 注意力机制
    C2fCIB,  # C2f + 紧凑倒置块
    C2fPSA,  # C2f + PSA 注意力
    C3Ghost,  # C3 + Ghost 瓶颈
    C3k2,  # C3 + 可定制卷积核
    C3x,  # C3 交叉卷积变体
    CBFuse,  # 跨分支特征融合模块
    CBLinear,  # 跨分支线性层
    ContrastiveHead,  # 对比学习头
    GhostBottleneck,  # Ghost 瓶颈块
    HGBlock,  # PPHGNetV2 HG 块
    HGStem,  # PPHGNetV2 Stem 块
    ImagePoolingAttn,  # 图像池化注意力
    MaxSigmoidAttnBlock,  # Max-Sigmoid 注意力块
    Proto,  # 分割原型生成模块
    RepC3,  # 重参数化 C3 块
    RepNCSPELAN4,  # 重参数化 NCSP-ELAN4 块
    RepVGGDW,  # RepVGG 深度可分离卷积
    ResNetLayer,  # ResNet 层
    SCDown,  # 可分离卷积下采样
    TorchVision,  # TorchVision 模型加载器
)

# 从 conv 模块导入各种卷积层
# 包括标准卷积、注意力卷积、深度可分离卷积等
from .conv import (
    CBAM,  # 卷积块注意力模块
    ChannelAttention,  # 通道注意力模块
    Concat,  # 张量拼接层
    Conv,  # 标准卷积 + 批归一化 + 激活
    Conv2,  # 双路卷积模块
    ConvTranspose,  # 转置卷积（上采样）
    DWConv,  # 深度可分离卷积
    DWConvTranspose2d,  # 深度可分离转置卷积
    Focus,  # 聚焦下采样模块
    GhostConv,  # Ghost 卷积（轻量级）
    Index,  # 索引选择层
    LightConv,  # 轻量级卷积
    RepConv,  # 重参数化卷积
    SpatialAttention,  # 空间注意力模块
)

# 从 head 模块导入各种任务头部
# 包括检测头、分割头、分类头、姿态估计头等
from .head import (
    OBB,  # 有向边界框检测头
    Classify,  # 分类头
    Detect,  # 标准检测头
    LRPCHead,  # 低秩参数化分类头
    Pose,  # 姿态估计头
    RTDETRDecoder,  # RT-DETR 解码器
    Segment,  # 实例分割头
    WorldDetect,  # 开放词汇检测头
    YOLOEDetect,  # YOLOE 检测头
    YOLOESegment,  # YOLOE 分割头
    v10Detect,  # YOLOv10 检测头
)

# 从 transformer 模块导入 Transformer 相关组件
# 包括自注意力、可变形注意力、MLP 等
from .transformer import (
    AIFI,  # 注意力-交互-前馈-迭代模块
    MLP,  # 多层感知机
    DeformableTransformerDecoder,  # 可变形 Transformer 解码器
    DeformableTransformerDecoderLayer,  # 可变形 Transformer 解码层
    LayerNorm2d,  # 2D 层归一化
    MLPBlock,  # MLP 块
    MSDeformAttn,  # 多尺度可变形注意力
    TransformerBlock,  # Transformer 块
    TransformerEncoderLayer,  # Transformer 编码层
    TransformerLayer,  # Transformer 层
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "OBB",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "A2C2f",
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
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Detect",
    "Focus",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    "SpatialAttention",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
)
