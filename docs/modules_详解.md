# Ultralytics NN Modules 详解

## 目录
1. [模块概述](#模块概述)
2. [conv.py - 卷积模块](#convpy---卷积模块)
3. [block.py - 块模块](#blockpy---块模块)
4. [head.py - 头部模块](#headpy---头部模块)
5. [transformer.py - Transformer 模块](#transformerpy---transformer-模块)
6. [activation.py - 激活函数模块](#activationpy---激活函数模块)
7. [utils.py - 工具函数模块](#utilspy---工具函数模块)
8. [模块关系图](#模块关系图)
9. [使用示例](#使用示例)

---

## 模块概述

`ultralytics/nn/modules/` 目录包含了 YOLO 系列模型的所有神经网络组件。这些模块按功能分为以下几类：

| 文件 | 主要功能 | 核心模块 |
|------|---------|---------|
| [conv.py](#convpy---卷积模块) | 基础卷积层 | Conv, DWConv, GhostConv, RepConv |
| [block.py](#blockpy---块模块) | 复合块结构 | C2f, C3, Bottleneck, SPPF, Attention |
| [head.py](#headpy---头部模块) | 任务头部 | Detect, Segment, Classify, Pose |
| [transformer.py](#transformerpy---transformer-模块) | Transformer 组件 | AIFI, MLP, MSDeformAttn |
| [activation.py](#activationpy---激活函数模块) | 激活函数 | SiLU, Hardswish, Mish |
| [utils.py](#utilspy---工具函数模块) | 工具函数 | 初始化、融合、量化等 |

---

## conv.py - 卷积模块

### 文件作用

提供 YOLO 模型中使用的各种卷积层和注意力机制，包括标准卷积、深度卷积、Ghost 卷积、重参数化卷积等。

### 工具函数

#### `autopad(k, p=None, d=1)`

**作用**: 自动计算 padding 以实现 'same' 形状输出

**参数**:
- `k` (int | tuple): 卷积核大小
- `p` (int | tuple, optional): 填充大小。如果为 None 则自动计算
- `d` (int): 膨胀率

**返回**: 计算得到的 padding 大小

**使用场景**: 在所有卷积模块中自动计算 padding

```python
# 示例
padding = autopad(k=3)  # 返回 1
padding = autopad(k=5)  # 返回 2
padding = autopad(k=3, d=2)  # 膨胀卷积，返回 2
```

---

### 核心类

#### 1. `Conv` - 标准卷积模块

**作用**: 标准卷积块，包含卷积 + 批归一化 + 激活函数

**结构**:
```
Input → Conv2d → BatchNorm2d → Activation → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 卷积核大小，默认 1
- `s` (int): 步长，默认 1
- `p` (int, optional): 填充大小
- `g` (int): 分组卷积的组数，默认 1
- `d` (int): 膨胀率，默认 1
- `act` (bool | nn.Module): 激活函数

**方法**:
- `forward(x)`: 标准前向传播（Conv → BN → Act）
- `forward_fuse(x)`: 融合模式前向传播（Conv → Act），用于推理加速

**使用场景**: YOLO 模型中最基础的卷积单元

```python
# 示例
conv = Conv(c1=64, c2=128, k=3, s=2)  # 输出尺寸减半
x = torch.randn(1, 64, 640, 640)
out = conv(x)  # shape: (1, 128, 320, 320)
```

---

#### 2. `Conv2` - 简化的 RepConv

**作用**: 包含额外 1×1 卷积分支的卷积模块

**结构**:
```
Input → Conv(3×3) → BN → Act → Output
  └──→ Conv(1×1) ──────────────┘ (融合)
```

**特点**:
- 训练时有两个分支（3×3 和 1×1）
- 推理时可融合为单个卷积

**使用场景**: 需要多分支特征的场景

---

#### 3. `LightConv` - 轻量级卷积

**作用**: 使用深度可分离卷积的轻量级模块

**结构**:
```
Input → Conv(1×1) → DWConv(k×k) → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): DWConv 卷积核大小，默认 1

**优势**: 参数量和计算量更少

```python
# 参数对比
standard_conv = Conv(64, 128, k=3)  # 参数量: 64×128×3×3 = 73,728
light_conv = LightConv(64, 128, k=3)  # 参数量: 64×128×1×1 + 128×3×3 = 9,344
```

---

#### 4. `DWConv` - 深度可分离卷积

**作用**: 深度卷积，每个输入通道独立卷积

**结构**:
```
Input → Conv2d(groups=c1) → BN → Act → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 卷积核大小，默认 1
- `s` (int): 步长，默认 1
- `d` (int): 膨胀率，默认 1
- `act` (bool | nn.Module): 激活函数

**特点**: `groups=c1`，实现深度卷积

**使用场景**: MobileNet 系列模型、轻量化网络

---

#### 5. `DWConvTranspose2d` - 深度可分离转置卷积

**作用**: 用于上采样的深度可分离转置卷积

**结构**:
```
Input → ConvTranspose2d(groups=c1) → BN → Act → Output
```

**使用场景**: 特征图上采样、解码器

---

#### 6. `ConvTranspose` - 转置卷积

**作用**: 标准转置卷积，用于上采样

**结构**:
```
Input → ConvTranspose2d → BN → Act → Output
```

**使用场景**: 特征图上采样、生成器网络

---

#### 7. `Focus` - 聚焦下采样

**作用**: 空间聚焦模块，无信息损失的下采样

**原理**:
```
输入: (B, C, H, W)
切片:
  x[::2, ::2, ...]  # 左上
  x[1::2, ::2, ...]  # 右上
  x[::2, 1::2, ...]  # 左下
  x[1::2, 1::2, ...]  # 右下
拼接: (B, 4C, H/2, W/2)
卷积: (B, C_out, H/2, W/2)
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 卷积核大小，默认 1
- `s` (int): 步长，默认 1
- `p` (int, optional): 填充
- `g` (int): 分组数，默认 1
- `act` (bool | nn.Module): 激活函数

**优势**: 保留所有信息，无下采样损失

**使用场景**: YOLOv5 的 Backbone 输入层

```python
# 示例
focus = Focus(c1=3, c2=64)
x = torch.randn(1, 3, 640, 640)
out = focus(x)  # shape: (1, 64, 320, 320)
```

---

#### 8. `GhostConv` - Ghost 卷积

**作用**: 轻量级卷积，使用廉价操作生成特征图

**原理**:
```
Input → Conv(c1, c_/2) → Output1
                       ↓
                  DWConv → Output2
Concat(Output1, Output2) → Final Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 卷积核大小，默认 1
- `s` (int): 步长，默认 1
- `g` (int): 分组数，默认 1
- `act` (bool | nn.Module): 激活函数

**优势**: 参数量约为标准卷积的一半

**使用场景**: GhostNet、轻量化 YOLO

```python
# 参数对比
standard = Conv(256, 256, k=3)  # 参数: 256×256×3×3 = 589,824
ghost = GhostConv(256, 256, k=3)  # 参数: ≈294,912 (约一半)
```

---

#### 9. `RepConv` - 重参数化卷积

**作用**: 训练时多分支，推理时单分支的卷积

**训练时结构**:
```
Input → Conv(3×3) → BN → Act ─┐
  ├──→ Conv(1×1) → BN ────────┤
  └──→ BN (Identity) ─────────┴→ Add → Output
```

**推理时结构**:
```
Input → Conv(3×3) → Act → Output  (融合后)
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int | tuple): 卷积核大小，默认 3
- `s` (int): 步长，默认 1
- `p` (int, optional): 填充
- `g` (int): 分组数，默认 1
- `d` (int | tuple): 膨胀率，默认 1
- `act` (bool | nn.Module): 激活函数
- `bn` (bool | nn.Module): 批归一化
- `deploy` (bool): 是否为部署模式

**方法**:
- `forward(x)`: 前向传播
- `forward_fuse(x)`: 融合后的前向传播
- `fuse_convs()`: 融合多个卷积分支
- `get_equivalent_kernel_bias()`: 获取等效的卷积核和偏置

**优势**:
- 训练时更好的特征表达
- 推理时更快的速度

**使用场景**: RepVGG、YOLOv6

```python
# 训练
repconv = RepConv(64, 128, k=3, deploy=False)
x = torch.randn(1, 64, 32, 32)
out = repconv(x)

# 推理前融合
repconv.fuse_convs()
out_fused = repconv(x)  # 更快
```

---

#### 10. `ChannelAttention` - 通道注意力

**作用**: 对通道维度进行注意力加权

**结构**:
```
Input → AdaptiveAvgPool → Conv(1×1) → ReLU → Conv(1×1) → Sigmoid → Weights
Input ────────────────────────────────────────────────────────────────× → Output
```

**参数**:
- `channels` (int): 输入通道数

**原理**: 学习每个通道的重要性权重

**使用场景**: CBAM、SE 模块

---

#### 11. `SpatialAttention` - 空间注意力

**作用**: 对空间维度进行注意力加权

**结构**:
```
Input → [AvgPool, MaxPool] → Concat → Conv(7×7) → Sigmoid → Weights
Input ──────────────────────────────────────────────────────────× → Output
```

**参数**:
- `kernel_size` (int): 卷积核大小，默认 7

**原理**: 学习每个空间位置的重要性权重

**使用场景**: CBAM 模块

---

#### 12. `CBAM` - 卷积块注意力模块

**作用**: 结合通道注意力和空间注意力

**结构**:
```
Input → ChannelAttention → SpatialAttention → Output
```

**参数**:
- `c1` (int): 输入通道数
- `kernel_size` (int): 空间注意力卷积核大小，默认 7

**优势**: 双重注意力机制，性能更好

**使用场景**: 需要注意力机制的场景

```python
# 示例
cbam = CBAM(c1=256, kernel_size=7)
x = torch.randn(1, 256, 32, 32)
out = cbam(x)  # shape: (1, 256, 32, 32)
```

---

#### 13. `Concat` - 张量拼接

**作用**: 在指定维度拼接多个张量

**参数**:
- `dimension` (int): 拼接维度，默认 1（通道维度）

**使用场景**: FPN/PAN 结构中的特征融合

```python
# 示例
concat = Concat(dimension=1)
x1 = torch.randn(1, 128, 32, 32)
x2 = torch.randn(1, 256, 32, 32)
out = concat([x1, x2])  # shape: (1, 384, 32, 32)
```

---

#### 14. `Index` - 索引选择

**作用**: 从张量列表中选择指定索引的张量

**参数**:
- `index` (int | list): 要选择的索引

**使用场景**: 多分支网络中的特征选择

---

## block.py - 块模块

### 文件作用

提供 YOLO 模型中使用的复合块结构，包括瓶颈块、CSP 块、注意力块、池化块等。这些模块是构建 Backbone 和 Head 的基本单元。

---

### 核心类

#### 1. `DFL` - 分布式焦点损失积分

**作用**: 将分布表示转换为边界框坐标

**原理**:
```
输入分布 → Softmax → 与位置编码加权求和 → 坐标值
```

**参数**:
- `c1` (int): 输入通道数，默认 16

**使用场景**: 检测头中的边界框回归

```python
# 示例
dfl = DFL(c1=16)
x = torch.randn(1, 16, 100)  # (batch, 16, num_anchors)
coords = dfl(x)  # shape: (1, 1, 100)
```

---

#### 2. `Proto` - 原型生成模块

**作用**: 生成实例分割的原型（prototype）

**结构**:
```
Input → Conv → Upsample → Conv → Upsample → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c_` (int): 中间通道数，默认 256
- `c2` (int): 输出通道数，默认 32

**使用场景**: YOLO 实例分割模型

---

#### 3. `Bottleneck` - 标准瓶颈块

**作用**: 残差瓶颈块，基础构建单元

**结构**:
```
Input → Conv(1×1) → Conv(3×3) → Output
  └─────────(shortcut)──────────┘
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `shortcut` (bool): 是否使用快捷连接，默认 True
- `g` (int): 分组数，默认 1
- `k` (int | tuple): 卷积核大小，默认 (3, 3)
- `e` (float): 扩展比例，默认 0.5

**使用场景**: ResNet、YOLOv5/v8 Backbone

```python
# 示例
bottleneck = Bottleneck(c1=128, c2=128, shortcut=True)
x = torch.randn(1, 128, 32, 32)
out = bottleneck(x)  # shape: (1, 128, 32, 32)
```

---

#### 4. `BottleneckCSP` - CSP 瓶颈块

**作用**: CSP（Cross Stage Partial）结构的瓶颈块

**结构**:
```
Input → Conv ─┬→ [Bottleneck × n] → Conv ─┐
              └→ Conv ────────────────────┴→ Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): Bottleneck 重复次数，默认 1
- `shortcut` (bool): Bottleneck 是否使用快捷连接，默认 True
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**优势**: 减少计算量，增强梯度流

**使用场景**: CSPNet、YOLOv4

---

#### 5. `C2f` - CSP 快速实现（2 卷积）

**作用**: YOLOv8 的核心模块，CSP 结构的快速实现

**结构**:
```
Input → Conv ─┬→ Split ─┬→ [Bottleneck] → ... → Concat → Conv → Output
              │         └→ ─────────────────────┘
              └→ ────────────────────────────────┘
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): Bottleneck 数量，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 False
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**特点**:
- 比 C3 更快
- 更灵活的梯度流

**使用场景**: YOLOv8 Backbone 和 Neck

```python
# 示例
c2f = C2f(c1=256, c2=256, n=3, shortcut=True)
x = torch.randn(1, 256, 32, 32)
out = c2f(x)  # shape: (1, 256, 32, 32)
```

---

#### 6. `C3` - CSP 瓶颈块（3 卷积）

**作用**: YOLOv5 的核心模块

**结构**:
```
Input → Conv ─┬→ [Bottleneck × n] → Conv ─┐
              └→ Conv ────────────────────┴→ Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): Bottleneck 数量，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 True
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**使用场景**: YOLOv5 Backbone 和 Neck

---

#### 7. `C3k2` - 可定制卷积核的 C3

**作用**: 支持自定义卷积核大小的 C3 模块

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): 块数量，默认 1
- `c3k` (bool): 是否使用 C3k 块，默认 False
- `e` (float): 扩展比例，默认 0.5
- `g` (int): 分组数，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 True

**使用场景**: YOLOv8m/l/x 模型

---

#### 8. `C3x` - C3 交叉卷积变体

**作用**: 使用交叉卷积的 C3 变体

**特点**: 更大的感受野

**使用场景**: 需要更大感受野的场景

---

#### 9. `C3TR` - C3 + Transformer

**作用**: 在 C3 中集成 Transformer 块

**结构**:
```
Input → Conv ─┬→ [TransformerBlock × n] ─┐
              └→ Conv ──────────────────┴→ Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): Transformer 块数量，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 True
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**使用场景**: 需要全局感受野的场景

---

#### 10. `C3Ghost` - C3 + Ghost 瓶颈

**作用**: 使用 Ghost 瓶颈的轻量化 C3

**优势**: 参数量更少

**使用场景**: 轻量化模型

---

#### 11. `SPP` - 空间金字塔池化

**作用**: 多尺度池化，增强感受野

**结构**:
```
Input → Conv ─┬→ MaxPool(5×5) ─┐
              ├→ MaxPool(9×9) ─┤
              ├→ MaxPool(13×13)┤
              └→ Identity ──────┴→ Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int | tuple): 池化核大小，默认 (5, 9, 13)

**使用场景**: YOLOv3-SPP、YOLOv5

```python
# 示例
spp = SPP(c1=1024, c2=1024, k=(5, 9, 13))
x = torch.randn(1, 1024, 20, 20)
out = spp(x)  # shape: (1, 1024, 20, 20)
```

---

#### 12. `SPPF` - 快速空间金字塔池化

**作用**: SPP 的快速版本

**结构**:
```
Input → Conv → MaxPool(5×5) → MaxPool(5×5) → MaxPool(5×5) → Concat → Conv → Output
              └─────────────┴─────────────┴─────────────┘
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 池化核大小，默认 5

**优势**:
- 计算量更少
- 速度更快
- 效果与 SPP 相当

**使用场景**: YOLOv5、YOLOv8

```python
# 速度对比
spp = SPP(1024, 1024, k=(5, 9, 13))
sppf = SPPF(1024, 1024, k=5)
# SPPF 速度约为 SPP 的 2 倍
```

---

#### 13. `C1` - CSP 单卷积瓶颈块

**作用**: 简化的 CSP 块，只有一个卷积

**使用场景**: 轻量化网络

---

#### 14. `C2` - CSP 双卷积瓶颈块

**作用**: 两个卷积的 CSP 块

**使用场景**: 中等复杂度网络

---

#### 15. `C2fAttn` - C2f + 注意力

**作用**: 在 C2f 中集成注意力机制

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): 块数量，默认 1
- `ec` (int): 嵌入通道数，默认 128
- `nh` (int): 注意力头数，默认 1
- `gc` (int): GC 通道数，默认 512
- `shortcut` (bool): 是否使用快捷连接，默认 False
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**使用场景**: 需要注意力机制的场景

---

#### 16. `Attention` - 多头自注意力

**作用**: 标准的多头自注意力机制

**参数**:
- `dim` (int): 输入维度
- `num_heads` (int): 注意力头数，默认 8
- `attn_ratio` (float): 注意力比例，默认 0.5

**结构**:
```
Input → QKV → MultiHeadAttention → Proj → Output
                                ↑
                                PE (位置编码)
```

**使用场景**: Transformer 模块、注意力增强

---

#### 17. `PSA` - 位置敏感注意力

**作用**: 位置敏感的注意力模块

**结构**:
```
Input → Conv ─┬→ Attention → FFN ─┐
              └→ ──────────────────┴→ Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `e` (float): 扩展比例，默认 0.5

**使用场景**: YOLOv8 高级版本

---

#### 18. `C2PSA` - C2 + PSA

**作用**: 在 C2 中集成 PSA 注意力

**使用场景**: 需要位置敏感注意力的场景

---

#### 19. `C2fPSA` - C2f + PSA

**作用**: 在 C2f 中集成 PSA 注意力

**使用场景**: YOLOv8 注意力增强版本

---

#### 20. `C2fCIB` - C2f + CIB

**作用**: 在 C2f 中使用紧凑倒置块

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): 块数量，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 False
- `lk` (bool): 是否使用大卷积核，默认 False
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5

**使用场景**: YOLOv9、YOLOv10

---

#### 21. `A2C2f` - 区域注意力 C2f

**作用**: 使用区域注意力的 C2f 模块

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): 块数量，默认 1
- `a2` (bool): 是否使用区域注意力，默认 True
- `area` (int): 区域数量，默认 1
- `residual` (bool): 是否使用残差连接，默认 False
- `mlp_ratio` (float): MLP 比例，默认 2.0
- `e` (float): 扩展比例，默认 0.5

**使用场景**: YOLOv9

---

#### 22. `RepC3` - 重参数化 C3

**作用**: 使用重参数化卷积的 C3

**优势**: 训练时多分支，推理时单分支

**使用场景**: YOLOv6

---

#### 23. `RepNCSPELAN4` - 重参数化 NCSP-ELAN4

**作用**: 复杂的重参数化 ELAN 块

**使用场景**: YOLOv7、YOLOv9

---

#### 24. `ADown` - 自适应下采样

**作用**: 使用平均池化和最大池化的下采样

**结构**:
```
Input → AvgPool ─┬→ Conv(3×3) ────┐
                 └→ MaxPool → Conv(1×1) ─┴→ Concat → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数

**使用场景**: YOLOv9

---

#### 25. `AConv` - 自适应卷积

**作用**: 结合平均池化的卷积

**使用场景**: YOLOv9

---

#### 26. `SPPELAN` - SPP-ELAN

**作用**: 结合 SPP 和 ELAN 的模块

**结构**:
```
Input → Conv → [MaxPool × 3] → Concat → Conv → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `c3` (int): 中间通道数
- `k` (int): 池化核大小，默认 5

**使用场景**: YOLOv9

---

#### 27. `CBLinear` - 跨分支线性层

**作用**: 将输入映射到多个输出分支

**参数**:
- `c1` (int): 输入通道数
- `c2s` (list[int]): 输出通道数列表
- `k` (int): 卷积核大小，默认 1
- `s` (int): 步长，默认 1
- `p` (int, optional): 填充
- `g` (int): 分组数，默认 1

**使用场景**: YOLOv10

---

#### 28. `CBFuse` - 跨分支融合

**作用**: 融合多个分支的特征

**参数**:
- `idx` (list[int]): 要融合的分支索引

**使用场景**: YOLOv10

---

#### 29. `RepVGGDW` - RepVGG 深度可分离卷积

**作用**: 重参数化的深度可分离卷积

**使用场景**: RepVGG

---

#### 30. `CIB` - 紧凑倒置块

**作用**: 轻量化的倒置残差块

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `shortcut` (bool): 是否使用快捷连接，默认 True
- `e` (float): 扩展比例，默认 0.5
- `lk` (bool): 是否使用大卷积核，默认 False

**使用场景**: YOLOv9、YOLOv10

---

#### 31. `SCDown` - 可分离卷积下采样

**作用**: 使用可分离卷积的下采样

**结构**:
```
Input → Conv(1×1) → DWConv(k×k, s) → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `k` (int): 卷积核大小
- `s` (int): 步长

**使用场景**: YOLOv10

---

#### 32. `HGStem` - HGNetV2 Stem

**作用**: PPHGNetV2 的 Stem 模块

**使用场景**: RT-DETR

---

#### 33. `HGBlock` - HGNetV2 块

**作用**: PPHGNetV2 的基础块

**使用场景**: RT-DETR

---

#### 34. `ELAN1` - ELAN 块

**作用**: 高效层聚合网络块

**使用场景**: YOLOv7

---

#### 35. `ResNetLayer` - ResNet 层

**作用**: 标准 ResNet 层

**使用场景**: 使用 ResNet 作为 Backbone

---

#### 36. `MaxSigmoidAttnBlock` - Max-Sigmoid 注意力块

**作用**: 使用 Max 和 Sigmoid 的注意力

**使用场景**: 特殊注意力需求

---

#### 37. `C3k` - 可定制卷积核的 C3

**作用**: 支持自定义卷积核大小的 C3

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `n` (int): 块数量，默认 1
- `shortcut` (bool): 是否使用快捷连接，默认 True
- `g` (int): 分组数，默认 1
- `e` (float): 扩展比例，默认 0.5
- `k` (int): 卷积核大小，默认 3

**使用场景**: 需要大卷积核的场景

---

#### 38. `GhostBottleneck` - Ghost 瓶颈块

**作用**: 使用 Ghost 卷积的瓶颈块

**使用场景**: GhostNet

---

#### 39. `ContrastiveHead` - 对比学习头

**作用**: 对比学习任务的头部

**使用场景**: 自监督学习

---

#### 40. `BNContrastiveHead` - BN 对比学习头

**作用**: 带批归一化的对比学习头

**使用场景**: 自监督学习

---

#### 41. `TorchVision` - TorchVision 模型加载器

**作用**: 加载 TorchVision 预训练模型

**参数**:
- `model` (str): 模型名称
- `weights` (str): 权重名称，默认 "DEFAULT"
- `unwrap` (bool): 是否展开，默认 True
- `truncate` (int): 截断层数，默认 2
- `split` (bool): 是否分割输出，默认 False

**使用场景**: 使用 TorchVision 模型作为 Backbone

```python
# 示例
resnet50 = TorchVision("resnet50", weights="DEFAULT")
x = torch.randn(1, 3, 224, 224)
features = resnet50(x)
```

---

#### 42. `ImagePoolingAttn` - 图像池化注意力

**作用**: 基于图像池化的注意力

**使用场景**: RT-DETR

---

## head.py - 头部模块

### 文件作用

提供 YOLO 模型的各种任务头部，包括检测头、分割头、分类头、姿态估计头等。这些模块负责将 Backbone 提取的特征转换为最终的预测结果。

---

### 核心类

#### 1. `Detect` - 标准检测头

**作用**: YOLO 标准检测头，输出边界框和类别

**结构**:
```
Input (P3, P4, P5) → Conv → [Bbox + Class] → Output
```

**参数**:
- `nc` (int): 类别数，默认 80
- `ch` (tuple): 输入通道数列表

**属性**:
- `nl` (int): 检测层数（通常为 3）
- `reg_max` (int): DFL 通道数，默认 16
- `no` (int): 每个锚点的输出数 = nc + reg_max × 4
- `stride` (Tensor): 特征图步长 [8, 16, 32]

**方法**:
- `forward(x)`: 前向传播
- `bias_init()`: 初始化偏置

**输出**:
- 训练模式: (batch, anchors, 4 + nc)
- 推理模式: (batch, anchors, 4 + nc) 或 [boxes, scores]

**使用场景**: YOLOv8 目标检测

```python
# 示例
detect = Detect(nc=80, ch=(256, 512, 1024))
x = [
    torch.randn(1, 256, 80, 80),   # P3
    torch.randn(1, 512, 40, 40),   # P4
    torch.randn(1, 1024, 20, 20)   # P5
]
output = detect(x)
```

---

#### 2. `Segment` - 实例分割头

**作用**: YOLO 实例分割头，输出边界框、类别和掩码

**结构**:
```
Input (P3, P4, P5) → Detect → [Bbox + Class + Mask] → Output
                   ↓
              Proto → Mask Prototypes
```

**参数**:
- `nc` (int): 类别数，默认 80
- `nm` (int): 掩码原型数量，默认 32
- `npr` (int): 原型数量，默认 256
- `ch` (tuple): 输入通道数列表

**输出**:
- 训练模式: (boxes, masks)
- 推理模式: (boxes, masks, protos)

**使用场景**: YOLOv8-seg 实例分割

```python
# 示例
segment = Segment(nc=80, nm=32, ch=(256, 512, 1024))
x = [
    torch.randn(1, 256, 80, 80),
    torch.randn(1, 512, 40, 40),
    torch.randn(1, 1024, 20, 20)
]
output = segment(x)
```

---

#### 3. `Pose` - 姿态估计头

**作用**: YOLO 姿态估计头，输出边界框、类别和关键点

**参数**:
- `nc` (int): 类别数，默认 1（通常为人）
- `kpt_shape` (tuple): 关键点形状 (num_kpts, dim)，默认 (17, 3)
- `ch` (tuple): 输入通道数列表

**输出**:
- 边界框: (batch, anchors, 4)
- 类别: (batch, anchors, nc)
- 关键点: (batch, anchors, num_kpts × 3)  # x, y, visibility

**使用场景**: YOLOv8-pose 姿态估计

```python
# 示例 - COCO 17 关键点
pose = Pose(nc=1, kpt_shape=(17, 3), ch=(256, 512, 1024))
x = [
    torch.randn(1, 256, 80, 80),
    torch.randn(1, 512, 40, 40),
    torch.randn(1, 1024, 20, 20)
]
output = pose(x)
```

---

#### 4. `OBB` - 有向边界框检测头

**作用**: 有向边界框（Oriented Bounding Box）检测头

**输出**:
- 边界框: (batch, anchors, 5)  # x, y, w, h, angle
- 类别: (batch, anchors, nc)

**使用场景**: YOLOv8-obb 旋转目标检测

```python
# 示例
obb = OBB(nc=15, ch=(256, 512, 1024))  # DOTA 数据集
x = [
    torch.randn(1, 256, 80, 80),
    torch.randn(1, 512, 40, 40),
    torch.randn(1, 1024, 20, 20)
]
output = obb(x)
```

---

#### 5. `Classify` - 分类头

**作用**: 图像分类头

**结构**:
```
Input → AdaptiveAvgPool → Flatten → Dropout → Linear → Output
```

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 类别数
- `k` (int): 池化核大小，默认 1
- `s` (int): 池化步长，默认 1
- `p` (float): Dropout 概率，默认 None
- `g` (int): 分组数，默认 1

**输出**:
- (batch, num_classes)

**使用场景**: YOLOv8-cls 图像分类

```python
# 示例
classify = Classify(c1=1024, c2=1000)  # ImageNet
x = torch.randn(1, 1024, 7, 7)
output = classify(x)  # shape: (1, 1000)
```

---

#### 6. `RTDETRDecoder` - RT-DETR 解码器

**作用**: RT-DETR 模型的解码器

**特点**:
- 基于 Transformer 的解码器
- 端到端检测，无需 NMS

**参数**:
- `nc` (int): 类别数
- `ch` (tuple): 输入通道数列表
- `hd` (int): 隐藏维度，默认 256
- `nq` (int): 查询数量，默认 300
- `ndp` (int): 解码器点数，默认 4
- `nh` (int): 注意力头数，默认 8
- `ndl` (int): 解码器层数，默认 6

**使用场景**: RT-DETR 实时检测

---

#### 7. `WorldDetect` - 开放词汇检测头

**作用**: 支持开放词汇的检测头

**特点**:
- 支持零样本检测
- 可检测训练时未见过的类别

**使用场景**: YOLO-World 开放词汇检测

---

#### 8. `v10Detect` - YOLOv10 检测头

**作用**: YOLOv10 的检测头

**特点**:
- 端到端检测
- 无需 NMS

**使用场景**: YOLOv10

---

#### 9. `YOLOEDetect` - YOLOE 检测头

**作用**: YOLOE（提示驱动）检测头

**使用场景**: YOLOE

---

#### 10. `YOLOESegment` - YOLOE 分割头

**作用**: YOLOE（提示驱动）分割头

**使用场景**: YOLOE-Seg

---

#### 11. `LRPCHead` - 低秩参数化分类头

**作用**: 轻量化的分类头

**使用场景**: 轻量化分类模型

---

## transformer.py - Transformer 模块

### 文件作用

提供 Transformer 相关组件，包括自注意力、可变形注意力、MLP 等。这些模块用于构建基于 Transformer 的检测器（如 RT-DETR）。

---

### 核心类

#### 1. `TransformerLayer` - Transformer 层

**作用**: 标准 Transformer 层

**结构**:
```
Input → MultiheadAttention → Add & Norm → FFN → Add & Norm → Output
```

**参数**:
- `c` (int): 输入通道数
- `num_heads` (int): 注意力头数

**使用场景**: Transformer 编码器/解码器

---

#### 2. `TransformerBlock` - Transformer 块

**作用**: 多层 Transformer 的堆叠

**参数**:
- `c1` (int): 输入通道数
- `c2` (int): 输出通道数
- `num_heads` (int): 注意力头数
- `num_layers` (int): 层数

**使用场景**: ViT、DETR

---

#### 3. `MLP` - 多层感知机

**作用**: 标准 MLP 模块

**结构**:
```
Input → Linear → GELU → Dropout → Linear → Dropout → Output
```

**参数**:
- `in_features` (int): 输入特征数
- `hidden_features` (int): 隐藏特征数
- `out_features` (int): 输出特征数
- `drop` (float): Dropout 概率

**使用场景**: Transformer FFN、分类头

---

#### 4. `MLPBlock` - MLP 块

**作用**: MLP 块，包含归一化

**使用场景**: Vision Transformer

---

#### 5. `LayerNorm2d` - 2D 层归一化

**作用**: 适用于 2D 特征图的层归一化

**参数**:
- `num_channels` (int): 通道数

**使用场景**: ConvNeXt、Swin Transformer

---

#### 6. `MSDeformAttn` - 多尺度可变形注意力

**作用**: 可变形 DETR 的核心模块

**特点**:
- 多尺度特征
- 可变形采样点
- 高效的注意力机制

**参数**:
- `d_model` (int): 模型维度，默认 256
- `n_levels` (int): 特征层级数，默认 4
- `n_heads` (int): 注意力头数，默认 8
- `n_points` (int): 采样点数，默认 4

**使用场景**: Deformable DETR、RT-DETR

---

#### 7. `DeformableTransformerDecoderLayer` - 可变形 Transformer 解码层

**作用**: 可变形 DETR 的解码层

**结构**:
```
Input → Self-Attn → Cross-Attn (可变形) → FFN → Output
```

**使用场景**: Deformable DETR、RT-DETR

---

#### 8. `DeformableTransformerDecoder` - 可变形 Transformer 解码器

**作用**: 多层可变形解码层的堆叠

**参数**:
- `hidden_dim` (int): 隐藏维度
- `decoder_layer` (nn.Module): 解码层
- `num_layers` (int): 层数

**使用场景**: Deformable DETR、RT-DETR

---

#### 9. `AIFI` - 注意力-交互-前馈-迭代模块

**作用**: RT-DETR 的特征增强模块

**参数**:
- `c1` (int): 输入通道数
- `cm` (int): 注意力模块隐藏维度，默认 2048
- `num_heads` (int): 注意力头数，默认 8
- `num_layers` (int): 层数，默认 2
- `dropout` (float): Dropout 概率，默认 0.0
- `act` (nn.Module): 激活函数，默认 nn.GELU()

**使用场景**: RT-DETR

---

#### 10. `TransformerEncoderLayer` - Transformer 编码层

**作用**: 标准 Transformer 编码层

**使用场景**: Transformer 编码器

---

## activation.py - 激活函数模块

### 文件作用

提供各种激活函数的实现。

---

### 核心函数/类

#### 1. `SiLU` (Swish)

**公式**: `f(x) = x * sigmoid(x)`

**使用场景**: 大多数现代 CNN

---

#### 2. `Hardswish`

**公式**: `f(x) = x * ReLU6(x + 3) / 6`

**优势**: 计算效率高

**使用场景**: MobileNetV3

---

#### 3. `Mish`

**公式**: `f(x) = x * tanh(softplus(x))`

**优势**: 平滑、非单调

**使用场景**: YOLOv4

---

## utils.py - 工具函数模块

### 文件作用

提供神经网络模块的工具函数，包括权重初始化、模型融合、量化等。

---

### 核心函数

#### 1. `initialize_weights(model)`

**作用**: 初始化模型权重

**参数**:
- `model` (nn.Module): 要初始化的模型

**使用场景**: 模型创建后

---

#### 2. `fuse_conv_and_bn(conv, bn)`

**作用**: 融合卷积和批归一化层

**参数**:
- `conv` (nn.Conv2d): 卷积层
- `bn` (nn.BatchNorm2d): 批归一化层

**返回**: 融合后的卷积层

**优势**: 推理速度提升约 10-20%

**使用场景**: 模型推理前

```python
# 示例
conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
bn = nn.BatchNorm2d(128)

# 融合
fused_conv = fuse_conv_and_bn(conv, bn)
```

---

#### 3. `model_info(model, verbose=False, imgsz=640)`

**作用**: 打印模型信息

**参数**:
- `model` (nn.Module): 模型
- `verbose` (bool): 是否详细输出
- `imgsz` (int): 输入图像大小

**输出**: 参数量、GFLOPs、层数等

---

#### 4. `scale_img(img, ratio=1.0, same_shape=False)`

**作用**: 缩放图像

**使用场景**: 数据增强

---

#### 5. `copy_attr(a, b, include=(), exclude=())`

**作用**: 复制对象属性

**使用场景**: 模型加载

---

#### 6. `intersect_dicts(da, db, exclude=())`

**作用**: 字典交集

**使用场景**: 权重加载

---

## 模块关系图

```
ultralytics/nn/modules/
│
├── conv.py (基础卷积层)
│   ├── Conv ──────────────┐
│   ├── DWConv ────────────┤
│   ├── GhostConv ─────────┤─→ block.py (复合块)
│   ├── RepConv ───────────┤   ├── Bottleneck
│   ├── LightConv ─────────┤   ├── C2f
│   └── Focus ─────────────┘   ├── C3
│                              ├── SPPF
├── block.py (复合块)          ├── Attention
│   ├── C2f ───────────────┐   └── ...
│   ├── C3 ────────────────┤
│   ├── Bottleneck ────────┤─→ head.py (任务头部)
│   ├── SPPF ──────────────┤   ├── Detect
│   ├── Attention ─────────┤   ├── Segment
│   └── ... ───────────────┘   ├── Classify
│                              └── Pose
├── transformer.py
│   ├── AIFI ──────────────┐
│   ├── MLP ───────────────┤─→ head.py
│   ├── MSDeformAttn ──────┤   └── RTDETRDecoder
│   └── ... ───────────────┘
│
├── head.py (任务头部)
│   ├── Detect
│   ├── Segment
│   ├── Classify
│   ├── Pose
│   └── RTDETRDecoder
│
├── activation.py
│   ├── SiLU
│   ├── Hardswish
│   └── Mish
│
└── utils.py (工具函数)
    ├── initialize_weights
    ├── fuse_conv_and_bn
    └── model_info
```

---

## 使用示例

### 示例 1: 构建简单的检测模型

```python
import torch
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect

# Backbone
x = torch.randn(1, 3, 640, 640)

# Stem
x = Conv(3, 64, k=6, s=2, p=2)(x)  # (1, 64, 320, 320)

# Stage 1
x = Conv(64, 128, k=3, s=2)(x)  # (1, 128, 160, 160)
x = C2f(128, 128, n=3)(x)

# Stage 2
x = Conv(128, 256, k=3, s=2)(x)  # (1, 256, 80, 80)
x = C2f(256, 256, n=6)(x)
p3 = x

# Stage 3
x = Conv(256, 512, k=3, s=2)(x)  # (1, 512, 40, 40)
x = C2f(512, 512, n=6)(x)
p4 = x

# Stage 4
x = Conv(512, 1024, k=3, s=2)(x)  # (1, 1024, 20, 20)
x = C2f(1024, 1024, n=3)(x)
x = SPPF(1024, 1024, k=5)(x)
p5 = x

# Head
detect = Detect(nc=80, ch=(256, 512, 1024))
output = detect([p3, p4, p5])
```

---

### 示例 2: 使用注意力模块

```python
from ultralytics.nn.modules import C2fAttn

# C2f with attention
c2f_attn = C2fAttn(
    c1=256,
    c2=256,
    n=3,
    ec=128,  # 嵌入通道
    nh=4     # 注意力头数
)

x = torch.randn(1, 256, 40, 40)
out = c2f_attn(x)  # (1, 256, 40, 40)
```

---

### 示例 3: 模型融合加速

```python
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.utils import fuse_conv_and_bn

# 创建模型
conv = Conv(64, 128, k=3, s=1)

# 训练
# ... training code ...

# 推理前融合
conv.forward = conv.forward_fuse

# 或者手动融合
fused_conv = fuse_conv_and_bn(conv.conv, conv.bn)
```

---

### 示例 4: 自定义分类器

```python
from ultralytics.nn.modules import Conv, C2f, SPPF, Classify

class CustomClassifier(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Backbone
        self.stem = Conv(3, 64, k=7, s=2)
        self.stage1 = torch.nn.Sequential(
            Conv(64, 128, k=3, s=2),
            C2f(128, 128, n=3)
        )
        self.stage2 = torch.nn.Sequential(
            Conv(128, 256, k=3, s=2),
            C2f(256, 256, n=6)
        )
        self.stage3 = torch.nn.Sequential(
            Conv(256, 512, k=3, s=2),
            C2f(512, 512, n=6),
            SPPF(512, 512)
        )

        # Classifier head
        self.head = Classify(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

# 使用
model = CustomClassifier(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
out = model(x)  # (1, 1000)
```

---

### 示例 5: 轻量化模型

```python
from ultralytics.nn.modules import LightConv, GhostConv, DWConv

# 标准卷积
standard = Conv(64, 128, k=3)
# 参数量: 64 × 128 × 3 × 3 = 73,728

# 轻量化方案 1: Light Conv
light = LightConv(64, 128, k=3)
# 参数量: 64 × 128 × 1 × 1 + 128 × 3 × 3 = 9,344
# 减少约 87%

# 轻量化方案 2: Ghost Conv
ghost = GhostConv(64, 128, k=3)
# 参数量: 约 36,864
# 减少约 50%

# 轻量化方案 3: Depthwise Separable
dw = torch.nn.Sequential(
    DWConv(64, 64, k=3),
    Conv(64, 128, k=1)
)
# 参数量: 64 × 3 × 3 + 64 × 128 × 1 × 1 = 8,768
# 减少约 88%
```

---

## 总结

### 模块选择指南

| 场景 | 推荐模块 | 原因 |
|------|---------|------|
| 标准检测 | C2f, SPPF, Detect | 平衡速度和精度 |
| 轻量化 | LightConv, GhostConv, DWConv | 减少参数量 |
| 高精度 | C2fAttn, PSA, Attention | 增强特征表达 |
| 实例分割 | Segment, Proto | 专门的分割模块 |
| 姿态估计 | Pose | 关键点检测 |
| 旋转目标 | OBB | 有向边界框 |
| 端到端 | RTDETRDecoder, v10Detect | 无需 NMS |
| 大感受野 | SPPF, SPP, ELAN1 | 多尺度池化 |

---

### 性能对比

| 模块 | 参数量 | 速度 | 精度 | 使用场景 |
|------|--------|------|------|---------|
| Conv | 标准 | 标准 | 标准 | 通用 |
| LightConv | 低 | 快 | 中 | 轻量化 |
| GhostConv | 低 | 快 | 中 | 轻量化 |
| RepConv | 中 | 训练慢/推理快 | 高 | 高精度 |
| C2f | 中 | 快 | 高 | YOLOv8 |
| C3 | 中 | 中 | 高 | YOLOv5 |
| C2fAttn | 高 | 慢 | 很高 | 高精度 |
| SPPF | 中 | 快 | 高 | 增强感受野 |

---

### 开发建议

1. **原型开发**: 使用标准模块（Conv, C2f, Detect）
2. **轻量化**: 替换为 LightConv、GhostConv
3. **高精度**: 添加注意力模块（Attention, PSA）
4. **推理优化**: 使用 RepConv 并在推理前融合
5. **自定义**: 基于现有模块组合创新

---

**参考资源**:
- [Ultralytics 官方文档](https://docs.ultralytics.com)
- [GitHub 仓库](https://github.com/ultralytics/ultralytics)
- [论文列表](https://github.com/ultralytics/ultralytics#documentation)
