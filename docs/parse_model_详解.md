# parse_model 函数详解

## 目录
1. [函数概述](#函数概述)
2. [函数签名与参数](#函数签名与参数)
3. [执行流程总览](#执行流程总览)
4. [详细步骤解析](#详细步骤解析)
5. [模块类型处理](#模块类型处理)
6. [实战示例](#实战示例)
7. [常见问题](#常见问题)

---

## 函数概述

`parse_model` 是 YOLO 模型构建的核心函数，负责将 YAML 配置文件解析为 PyTorch 神经网络模型。

**位置**: `ultralytics/nn/tasks.py:1509`

**主要功能**:
- 读取 YAML 配置中的层定义
- 根据模型规模（n/s/m/l/x）调整网络深度和宽度
- 实例化各个模块并组装成完整网络
- 追踪需要保存的中间层输出（用于特征融合）

---

## 函数签名与参数

```python
def parse_model(d, ch, verbose=True):
    """
    将 YOLO model.yaml 字典解析为 PyTorch 模型

    参数:
        d (dict): 模型字典，包含以下关键字段:
            - 'nc': 类别数量
            - 'scales': 模型缩放参数 {n: [depth, width, max_ch], ...}
            - 'backbone': Backbone 层定义列表
            - 'head': Head 层定义列表
            - 'activation': 激活函数（可选）
            - 'depth_multiple': 深度乘数（可选）
            - 'width_multiple': 宽度乘数（可选）
        ch (int): 输入通道数，通常为 3（RGB 图像）
        verbose (bool): 是否打印模型详细信息

    返回:
        model (torch.nn.Sequential): PyTorch 顺序模型
        save (list): 需要保存输出的层索引列表（已排序）
    """
```

---

## 执行流程总览

```
┌─────────────────────────────────────────────┐
│ 1. 初始化参数                                 │
│    - 提取 nc, scales, depth, width           │
│    - 设置默认激活函数                          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 2. 准备数据结构                               │
│    - ch = [3]  (通道列表)                     │
│    - layers = []  (层列表)                    │
│    - save = []  (保存列表)                    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3. 遍历 backbone + head 每一层                │
│    for i, (f, n, m, args) in enumerate(...) │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼──────┐    ┌───────▼──────┐
│ 3.1 解析模块  │    │ 3.2 解析参数  │
│   - 获取类    │    │   - 替换变量  │
│   - 导入模块  │    │   - 计算通道  │
└───────┬──────┘    └───────┬──────┘
        │                   │
        └─────────┬─────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3.3 处理不同模块类型                          │
│    - base_modules: Conv, C2f, SPPF...       │
│    - Concat: 拼接层                          │
│    - Detect/Segment/Pose: 检测头             │
│    - 其他特殊模块                             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3.4 实例化模块                                │
│    m_ = m(*args) 或 Sequential(...)          │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 3.5 记录信息                                  │
│    - 添加到 layers                            │
│    - 更新 ch 列表                             │
│    - 添加到 save 列表                         │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ 4. 返回模型                                   │
│    return Sequential(*layers), sorted(save)  │
└─────────────────────────────────────────────┘
```

---

## 详细步骤解析

### 步骤 1: 初始化参数

```python
# 提取配置参数
nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))

# 获取模型规模
scale = d.get("scale")  # 'n', 's', 'm', 'l', 'x'
if scales:
    depth, width, max_channels = scales[scale]
    # 例如: scale='n' -> depth=0.33, width=0.25, max_channels=1024
```

**参数说明**:

| 参数 | 说明 | 示例值 (YOLOv8n) |
|------|------|------------------|
| `nc` | 类别数 | 80 |
| `depth` | 深度乘数 | 0.33 |
| `width` | 宽度乘数 | 0.25 |
| `max_channels` | 最大通道数 | 1024 |

**深度和宽度缩放**:
```python
# 深度缩放: 控制层的重复次数
n = max(round(n * depth), 1) if n > 1 else n

# 宽度缩放: 控制通道数
c2 = make_divisible(min(c2, max_channels) * width, 8)
```

### 步骤 2: 准备数据结构

```python
ch = [ch]              # ch = [3], 通道列表
layers = []            # 存储所有层
save = []              # 存储需要保存输出的层索引
c2 = ch[-1]            # 当前输出通道数
```

**通道追踪机制**:
- `ch` 列表记录每一层的输出通道数
- `ch[i]` 表示第 i 层的输出通道数
- `f` 参数通过索引 `ch` 来获取输入通道数

### 步骤 3: 遍历层定义

```python
for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
    # i: 当前层索引 (0, 1, 2, ...)
    # f: from 参数，输入来源 (-1, 6, [-1, 6], ...)
    # n: repeats 参数，重复次数 (1, 3, 6, ...)
    # m: module 参数，模块类型 ('Conv', 'C2f', 'Detect', ...)
    # args: 模块参数 ([64, 3, 2], [128, True], ...)
```

#### 3.1 解析模块类

```python
# 获取模块类
m = (
    getattr(torch.nn, m[3:])              # 如果是 'nn.Upsample'
    if "nn." in m
    else getattr(__import__("torchvision").ops, m[16:])  # 如果是 'torchvision.ops.xxx'
    if "torchvision.ops." in m
    else globals()[m]                      # 否则从全局命名空间获取
)
```

**示例**:
```python
# m = 'Conv' -> globals()['Conv'] -> <class 'Conv'>
# m = 'nn.Upsample' -> torch.nn.Upsample
# m = 'C2f' -> globals()['C2f'] -> <class 'C2f'>
```

#### 3.2 解析参数

```python
# 替换字符串参数为实际值
for j, a in enumerate(args):
    if isinstance(a, str):
        with contextlib.suppress(ValueError):
            # 如果 a 是局部变量名，使用其值
            # 否则尝试字面量求值（例如 'True' -> True）
            args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
```

**参数替换示例**:
```python
# YAML 中: args = [nc]
# 替换后: args = [80]

# YAML 中: args = [256, True]
# 替换后: args = [256, True]
```

#### 3.3 计算深度增益

```python
n = n_ = max(round(n * depth), 1) if n > 1 else n
```

**示例**:
```python
# YOLOv8n (depth=0.33):
# YAML 中 n=3 -> n = max(round(3 * 0.33), 1) = 1
# YAML 中 n=6 -> n = max(round(6 * 0.33), 1) = 2

# YOLOv8l (depth=1.0):
# YAML 中 n=3 -> n = 3
# YAML 中 n=6 -> n = 6
```

---

## 模块类型处理

### 类型 1: 基础模块 (base_modules)

包括: `Conv`, `C2f`, `SPPF`, `Bottleneck` 等

```python
if m in base_modules:
    # 1. 获取输入通道数
    c1 = ch[f]

    # 2. 获取输出通道数
    c2 = args[0]

    # 3. 应用宽度缩放
    if c2 != nc:  # 不是分类层
        c2 = make_divisible(min(c2, max_channels) * width, 8)

    # 4. 构造参数列表
    args = [c1, c2, *args[1:]]

    # 5. 插入重复次数（如果需要）
    if m in repeat_modules:
        args.insert(2, n)  # [c1, c2, n, ...]
        n = 1
```

**处理流程图**:
```
YAML: [-1, 3, C2f, [256, True]]
                ↓
f=-1, n=3, m=C2f, args=[256, True]
                ↓
c1 = ch[-1] = 128  (上一层输出)
c2 = 256
                ↓
应用宽度缩放:
c2 = make_divisible(256 * 0.25, 8) = 64
                ↓
计算深度:
n = max(round(3 * 0.33), 1) = 1
                ↓
构造参数:
args = [128, 64, 1, True]
                ↓
实例化:
m_ = C2f(c1=128, c2=64, n=1, shortcut=True)
```

### 类型 2: Concat 拼接层

```python
elif m is Concat:
    # 计算输出通道数 = 所有输入通道数之和
    c2 = sum(ch[x] for x in f)
```

**示例**:
```python
# YAML: [[-1, 6], 1, Concat, [1]]
# f = [-1, 6]
# ch[-1] = 512, ch[6] = 512
# c2 = 512 + 512 = 1024
```

### 类型 3: 检测头 (Detect/Segment/Pose)

```python
elif m in {Detect, Segment, Pose, OBB}:
    # 添加各个尺度的通道数列表
    args.append([ch[x] for x in f])

    # 对于分割任务，调整原型通道数
    if m is Segment:
        args[2] = make_divisible(min(args[2], max_channels) * width, 8)
```

**示例**:
```python
# YAML: [[15, 18, 21], 1, Detect, [nc]]
# f = [15, 18, 21]
# ch[15] = 256, ch[18] = 512, ch[21] = 1024
# args = [80, [256, 512, 1024]]
# m_ = Detect(nc=80, ch=[256, 512, 1024])
```

### 类型 4: 上采样层

```python
# YAML: [-1, 1, nn.Upsample, [None, 2, 'nearest']]
# m = torch.nn.Upsample
# args = [None, 2, 'nearest']
# m_ = nn.Upsample(size=None, scale_factor=2, mode='nearest')
```

### 类型 5: AIFI (注意力模块)

```python
elif m is AIFI:
    args = [ch[f], *args]
    # args = [输入通道数, 其他参数...]
```

---

## 步骤 4: 实例化模块

```python
# 如果 n > 1，创建 Sequential 容器包含 n 个相同模块
# 否则创建单个模块
m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
```

**示例**:
```python
# n = 1
m_ = Conv(64, 128, 3, 2)

# n = 3
m_ = Sequential(
    C2f(128, 256, 1, True),
    C2f(128, 256, 1, True),
    C2f(128, 256, 1, True)
)
```

---

## 步骤 5: 记录信息

```python
# 1. 提取模块类型名称
t = str(m)[8:-2].replace("__main__.", "")

# 2. 计算参数数量
m_.np = sum(x.numel() for x in m_.parameters())

# 3. 附加元数据
m_.i = i        # 层索引
m_.f = f        # from 参数
m_.type = t     # 模块类型

# 4. 打印信息（如果 verbose=True）
if verbose:
    LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")

# 5. 更新保存列表
# 如果其他层引用了这一层，将其索引添加到 save
save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

# 6. 添加到层列表
layers.append(m_)

# 7. 更新通道列表
if i == 0:
    ch = []  # 清空初始通道
ch.append(c2)    # 添加当前层输出通道
```

**保存列表机制**:
```python
# 示例: f = [-1, 6]
# 提取出 6，表示需要保存第 6 层的输出
save.extend([6])

# 最终 save = [4, 6, 9, 12, 15, 18, 21] (排序后)
```

---

## 步骤 6: 返回模型

```python
return torch.nn.Sequential(*layers), sorted(save)
```

**返回值**:
- `model`: 包含所有层的 Sequential 模型
- `save`: 需要保存输出的层索引列表（用于前向传播时的特征融合）

---

## 实战示例

### 示例 1: 简单卷积层

**YAML 定义**:
```yaml
- [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
```

**解析过程**:
```python
i = 0
f = -1              # 输入来自上一层
n = 1               # 不重复
m = Conv            # 模块类
args = [64, 3, 2]   # [输出通道, 卷积核, 步长]

# 步骤 1: 获取输入通道
c1 = ch[-1] = 3  # 输入图像

# 步骤 2: 应用宽度缩放
c2 = make_divisible(64 * 0.25, 8) = 16  # YOLOv8n

# 步骤 3: 构造参数
args = [3, 16, 3, 2]  # [c1, c2, k, s]

# 步骤 4: 实例化
m_ = Conv(c1=3, c2=16, k=3, s=2)

# 步骤 5: 更新状态
ch.append(16)
layers.append(m_)
```

### 示例 2: C2f 模块

**YAML 定义**:
```yaml
- [-1, 3, C2f, [256, True]]  # 4
```

**解析过程**:
```python
i = 4
f = -1
n = 3
m = C2f
args = [256, True]

# 步骤 1: 获取输入通道
c1 = ch[-1] = 128

# 步骤 2: 应用宽度缩放
c2 = make_divisible(256 * 0.25, 8) = 64

# 步骤 3: 计算深度
n = max(round(3 * 0.33), 1) = 1

# 步骤 4: 构造参数（插入 n）
args = [128, 64, 1, True]  # [c1, c2, n, shortcut]

# 步骤 5: 实例化
m_ = C2f(c1=128, c2=64, n=1, shortcut=True)

# 步骤 6: 更新状态
ch.append(64)
layers.append(m_)
```

### 示例 3: Concat 层

**YAML 定义**:
```yaml
- [[-1, 6], 1, Concat, [1]]  # 11
```

**解析过程**:
```python
i = 11
f = [-1, 6]          # 从第 10 层和第 6 层获取输入
n = 1
m = Concat
args = [1]           # 拼接维度

# 步骤 1: 计算输出通道
c2 = ch[-1] + ch[6] = 512 + 512 = 1024

# 步骤 2: 实例化
m_ = Concat(dimension=1)

# 步骤 3: 更新保存列表
save.extend([6])  # 需要保存第 6 层的输出

# 步骤 4: 更新状态
ch.append(1024)
layers.append(m_)
```

### 示例 4: Detect 检测头

**YAML 定义**:
```yaml
- [[15, 18, 21], 1, Detect, [nc]]  # 22
```

**解析过程**:
```python
i = 22
f = [15, 18, 21]     # 三个尺度的特征图
n = 1
m = Detect
args = [80]          # nc=80

# 步骤 1: 添加各尺度的通道数
args.append([ch[15], ch[18], ch[21]])
args = [80, [256, 512, 1024]]

# 步骤 2: 实例化
m_ = Detect(nc=80, ch=[256, 512, 1024])

# 步骤 3: 更新保存列表
save.extend([15, 18, 21])

# 步骤 4: 更新状态
ch.append(80)  # 输出通道
layers.append(m_)
```

---

## 完整示例: YOLOv8n 构建过程

### 输入配置

```yaml
nc: 80
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  - [-1, 3, C2f, [128, True]]       # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  - [-1, 6, C2f, [256, True]]       # 4

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 5
  - [[-1, 3], 1, Concat, [1]]                   # 6
  - [-1, 3, C2f, [256]]                         # 7
  - [[7], 1, Detect, [nc]]                      # 8
```

### 逐层解析

| 层 | from | n | module | args | c1 | c2 (缩放前) | c2 (缩放后) | n (缩放后) | 实例化 |
|---|------|---|--------|------|----|-----------|-----------|-----------|----|
| 0 | -1 | 1 | Conv | [64, 3, 2] | 3 | 64 | 16 | 1 | Conv(3, 16, 3, 2) |
| 1 | -1 | 1 | Conv | [128, 3, 2] | 16 | 128 | 32 | 1 | Conv(16, 32, 3, 2) |
| 2 | -1 | 3 | C2f | [128, True] | 32 | 128 | 32 | 1 | C2f(32, 32, 1, True) |
| 3 | -1 | 1 | Conv | [256, 3, 2] | 32 | 256 | 64 | 1 | Conv(32, 64, 3, 2) |
| 4 | -1 | 6 | C2f | [256, True] | 64 | 256 | 64 | 2 | C2f(64, 64, 2, True) |
| 5 | -1 | 1 | Upsample | [None, 2, 'nearest'] | 64 | - | 64 | 1 | Upsample(None, 2, 'nearest') |
| 6 | [-1, 3] | 1 | Concat | [1] | - | - | 128 | 1 | Concat(1) |
| 7 | -1 | 3 | C2f | [256] | 128 | 256 | 64 | 1 | C2f(128, 64, 1) |
| 8 | [7] | 1 | Detect | [80] | - | - | 80 | 1 | Detect(80, [64]) |

### 通道列表变化

```python
初始: ch = [3]

层 0: ch = [16]        # Conv 输出 16 通道
层 1: ch = [16, 32]    # Conv 输出 32 通道
层 2: ch = [16, 32, 32]     # C2f 输出 32 通道
层 3: ch = [16, 32, 32, 64]      # Conv 输出 64 通道
层 4: ch = [16, 32, 32, 64, 64]       # C2f 输出 64 通道
层 5: ch = [16, 32, 32, 64, 64, 64]        # Upsample 输出 64 通道
层 6: ch = [16, 32, 32, 64, 64, 64, 128]   # Concat 输出 128 通道
层 7: ch = [16, 32, 32, 64, 64, 64, 128, 64]    # C2f 输出 64 通道
层 8: ch = [16, 32, 32, 64, 64, 64, 128, 64, 80] # Detect 输出 80 通道
```

### 保存列表

```python
save = [3, 7]  # 需要保存第 3 层和第 7 层的输出
# 第 3 层被第 6 层的 Concat 引用
# 第 7 层被第 8 层的 Detect 引用
```

---

## 核心机制详解

### 1. 通道追踪机制

**目的**: 自动计算每一层的输入输出通道数

**实现**:
```python
# ch 列表记录每一层的输出通道数
ch = [3]  # 初始输入

# 第 i 层
c1 = ch[f]  # 输入通道 = 第 f 层的输出通道
c2 = ...    # 计算输出通道
ch.append(c2)  # 记录输出通道
```

**示例**:
```python
# 层 0: Conv(3, 16, 3, 2)
ch = [3, 16]

# 层 1: Conv(16, 32, 3, 2)
ch = [3, 16, 32]

# 层 2: C2f(32, 32, 1, True)
ch = [3, 16, 32, 32]
```

### 2. 保存列表机制

**目的**: 标记哪些层的输出需要保存（用于后续层的特征融合）

**实现**:
```python
# 提取 from 参数中非 -1 的索引
save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
```

**示例**:
```python
# 层 6: f = [-1, 3]
# 提取: [3]
save.extend([3])

# 层 8: f = [7]
# 提取: [7]
save.extend([7])

# 最终: save = [3, 7]
```

### 3. 深度宽度缩放机制

**深度缩放** (控制层数):
```python
n = max(round(n * depth), 1) if n > 1 else n
```

| 模型 | depth | YAML n=3 | 实际 n | YAML n=6 | 实际 n |
|------|-------|----------|--------|----------|--------|
| YOLOv8n | 0.33 | 3 | 1 | 6 | 2 |
| YOLOv8s | 0.33 | 3 | 1 | 6 | 2 |
| YOLOv8m | 0.67 | 3 | 2 | 6 | 4 |
| YOLOv8l | 1.00 | 3 | 3 | 6 | 6 |
| YOLOv8x | 1.00 | 3 | 3 | 6 | 6 |

**宽度缩放** (控制通道数):
```python
c2 = make_divisible(min(c2, max_channels) * width, 8)
```

| 模型 | width | YAML c2=256 | 实际 c2 | YAML c2=512 | 实际 c2 |
|------|-------|-------------|---------|-------------|---------|
| YOLOv8n | 0.25 | 256 | 64 | 512 | 128 |
| YOLOv8s | 0.50 | 256 | 128 | 512 | 256 |
| YOLOv8m | 0.75 | 256 | 192 | 512 | 384 |
| YOLOv8l | 1.00 | 256 | 256 | 512 | 512 |
| YOLOv8x | 1.25 | 256 | 320 | 512 | 640 |

### 4. make_divisible 函数

**目的**: 确保通道数是 8 的倍数（GPU 优化）

```python
def make_divisible(x, divisor=8):
    """返回最接近 x 的能被 divisor 整除的数"""
    return math.ceil(x / divisor) * divisor
```

**示例**:
```python
make_divisible(64 * 0.25, 8) = make_divisible(16, 8) = 16
make_divisible(256 * 0.25, 8) = make_divisible(64, 8) = 64
make_divisible(512 * 0.25, 8) = make_divisible(128, 8) = 128
```

---

## 常见问题

### Q1: 为什么通道数要是 8 的倍数？

**答**: GPU 优化。现代 GPU（特别是 NVIDIA Tensor Cores）在处理 8 的倍数的张量时性能最佳。

### Q2: from 参数的负数索引如何工作？

**答**:
```python
-1: 上一层 (等价于 i-1)
-2: 上上层 (等价于 i-2)
6: 第 6 层 (绝对索引)
[-1, 6]: 第 i-1 层和第 6 层
```

### Q3: 为什么有些层的 n > 1？

**答**: 重复模块以增加网络深度。例如:
```python
# n = 3 表示堆叠 3 个相同的模块
m_ = Sequential(
    Bottleneck(...),
    Bottleneck(...),
    Bottleneck(...)
)
```

### Q4: save 列表的作用是什么？

**答**: 标记需要保存中间输出的层，用于特征融合（FPN/PAN）。在前向传播时，这些层的输出会被缓存供后续层使用。

### Q5: 如何添加自定义模块？

**答**:
1. 在 `ultralytics/nn/modules/` 中定义模块类
2. 在 `ultralytics/nn/modules/__init__.py` 中导入
3. 在 `ultralytics/nn/tasks.py` 中添加到相应的模块集合
4. 在 YAML 中使用模块名称

**示例**:
```python
# 1. 定义模块
class MyModule(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = Conv(c1, c2, k)

    def forward(self, x):
        return self.conv(x)

# 2. 导入
from .block import MyModule

# 3. 添加到 base_modules
base_modules = frozenset({..., MyModule})

# 4. 在 YAML 中使用
- [-1, 1, MyModule, [64, 3]]
```

### Q6: 如何调试模型构建过程？

**答**:
```python
# 方法 1: 使用 verbose=True
model, save = parse_model(d, ch=3, verbose=True)

# 方法 2: 打印模型结构
print(model)

# 方法 3: 查看每层的输出形状
x = torch.randn(1, 3, 640, 640)
for i, layer in enumerate(model):
    x = layer(x)
    print(f"Layer {i}: {x.shape}")
```

### Q7: 不同模型规模的参数对比

| 模型 | depth | width | 参数量 | GFLOPs |
|------|-------|-------|--------|--------|
| YOLOv8n | 0.33 | 0.25 | 3.2M | 8.7 |
| YOLOv8s | 0.33 | 0.50 | 11.2M | 28.6 |
| YOLOv8m | 0.67 | 0.75 | 25.9M | 78.9 |
| YOLOv8l | 1.00 | 1.00 | 43.7M | 165.2 |
| YOLOv8x | 1.00 | 1.25 | 68.2M | 257.8 |

---

## 总结

`parse_model` 函数是 YOLO 模型构建的核心，其主要特点：

1. **配置驱动**: 通过 YAML 文件定义网络结构
2. **自动缩放**: 根据模型规模自动调整深度和宽度
3. **通道追踪**: 自动计算每层的输入输出通道数
4. **特征融合**: 通过 save 列表管理中间层输出
5. **模块化设计**: 易于扩展和自定义

**关键流程**:
```
YAML 配置 → 解析层定义 → 缩放参数 → 实例化模块 → 组装模型
```

**核心机制**:
- 深度缩放: `n = round(n * depth)`
- 宽度缩放: `c2 = make_divisible(c2 * width, 8)`
- 通道追踪: `ch` 列表
- 特征保存: `save` 列表

通过理解 `parse_model` 的工作原理，你可以：
- 自定义网络架构
- 添加新的模块类型
- 调整模型规模
- 优化模型性能

---

**参考资源**:
- [YOLOv8 官方文档](https://docs.ultralytics.com)
- [模型配置示例](../cfg/models/)
- [模块定义](../nn/modules/)
