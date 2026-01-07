# YOLO 网络结构 YAML 配置文档

## 目录
1. [概述](#概述)
2. [YAML 文件结构](#yaml-文件结构)
3. [全局参数配置](#全局参数配置)
4. [层定义语法](#层定义语法)
5. [Backbone 配置](#backbone-配置)
6. [Head 配置](#head-配置)
7. [常用模块说明](#常用模块说明)
8. [实战示例](#实战示例)
9. [高级技巧](#高级技巧)
10. [附录](#附录)

---

## 概述

YOLOv8 使用 YAML 文件来定义神经网络结构，这种方式具有以下优势：

- **可读性强**：清晰的层次结构，易于理解和修改
- **灵活性高**：支持快速调整网络架构
- **模块化设计**：便于复用和组合不同的网络组件
- **版本控制友好**：纯文本格式，方便 Git 管理

---

## YAML 文件结构

一个完整的 YOLO 配置文件包含三个主要部分：

```yaml
# 1. 全局参数
nc: 80                    # 类别数量
scales: {...}             # 模型缩放参数

# 2. Backbone（骨干网络）
backbone:
  - [层定义1]
  - [层定义2]
  - ...

# 3. Head（检测头）
head:
  - [层定义1]
  - [层定义2]
  - ...
```

---

## 全局参数配置

### nc (Number of Classes)
```yaml
nc: 80  # 检测的类别数量
```

### scales（模型缩放参数）
```yaml
scales:
  # [depth_multiple, width_multiple, max_channels]
  n: [0.33, 0.25, 1024]  # nano 版本
  s: [0.33, 0.50, 1024]  # small 版本
  m: [0.67, 0.75, 768]   # medium 版本
  l: [1.00, 1.00, 512]   # large 版本
  x: [1.00, 1.25, 512]   # xlarge 版本
```

**参数说明**：
- `depth_multiple`：控制网络深度（repeats 参数的乘数）
- `width_multiple`：控制网络宽度（通道数的乘数）
- `max_channels`：最大通道数限制

---

## 层定义语法

每一层使用以下格式定义：

```yaml
[from, repeats, module, args]
```

### 参数详解

| 参数 | 说明 | 示例 |
|------|------|------|
| `from` | 输入来源层索引 | `-1` (上一层), `6` (第6层), `[-1, 6]` (多个层) |
| `repeats` | 模块重复次数 | `1`, `3`, `6` |
| `module` | 模块类型 | `Conv`, `C2f`, `SPPF`, `Detect` |
| `args` | 模块参数 | `[64, 3, 2]` (channels, kernel, stride) |

### 示例解析

```yaml
- [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
```

**含义**：
- `-1`：输入来自上一层
- `1`：该模块不重复（只出现1次）
- `Conv`：使用卷积模块
- `[64, 3, 2]`：输出64通道，3×3卷积核，步长2

---

## Backbone 配置

Backbone 负责特征提取，通常采用逐步下采样的结构。

### YOLOv8 Backbone 结构

```yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2   (640 -> 320)
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4   (320 -> 160)
  - [-1, 3, C2f, [128, True]]       # 2        特征提取
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8   (160 -> 80)
  - [-1, 6, C2f, [256, True]]       # 4        特征提取
  - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16  (80 -> 40)
  - [-1, 6, C2f, [512, True]]       # 6        特征提取
  - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32  (40 -> 20)
  - [-1, 3, C2f, [1024, True]]      # 8        特征提取
  - [-1, 1, SPPF, [1024, 5]]        # 9        空间金字塔池化
```

### 特征层说明

| 层标记 | 分辨率 | 特征尺度 | 用途 |
|--------|--------|----------|------|
| P1/2 | 320×320 | 大目标 | 仅在 Backbone |
| P2/4 | 160×160 | 大目标 | 仅在 Backbone |
| P3/8 | 80×80 | 小目标检测 | 输出层 |
| P4/16 | 40×40 | 中等目标检测 | 输出层 |
| P5/32 | 20×20 | 大目标检测 | 输出层 |

---

## Head 配置

Head 采用 FPN（特征金字塔网络）+ PAN（路径聚合网络）结构。

### YOLOv8 Head 结构

```yaml
head:
  # 上采样路径（FPN）
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10: P5 上采样
  - [[-1, 6], 1, Concat, [1]]                   # 11: 融合 P4
  - [-1, 3, C2f, [512]]                         # 12: 特征处理

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13: P4 上采样
  - [[-1, 4], 1, Concat, [1]]                   # 14: 融合 P3
  - [-1, 3, C2f, [256]]                         # 15: P3/8-small

  # 下采样路径（PAN）
  - [-1, 1, Conv, [256, 3, 2]]                  # 16: 下采样
  - [[-1, 12], 1, Concat, [1]]                  # 17: 融合 P4
  - [-1, 3, C2f, [512]]                         # 18: P4/16-medium

  - [-1, 1, Conv, [512, 3, 2]]                  # 19: 下采样
  - [[-1, 9], 1, Concat, [1]]                   # 20: 融合 P5
  - [-1, 3, C2f, [1024]]                        # 21: P5/32-large

  # 检测头
  - [[15, 18, 21], 1, Detect, [nc]]             # 22: 多尺度检测
```

### 数据流向图

```
Backbone输出:
  P3(4) ──┐
  P4(6) ──┼──┐
  P5(9) ──┼──┼──> Head处理
          │  │
FPN上采样 │  │
  ↓       │  │
  P5 -> P4(融合6) -> P3(融合4)
                      │
PAN下采样              │
  ↓                   ↓
  P3 -> P4(融合12) -> P5(融合9)
  │      │            │
  ↓      ↓            ↓
 小目标  中目标       大目标
(15)    (18)         (21)
  └──────┴────────────┘
         Detect
```

---

## 常用模块说明

### Conv（标准卷积）
```yaml
- [-1, 1, Conv, [channels, kernel_size, stride, padding]]
```
- 包含：Conv2d + BatchNorm + SiLU 激活

### C2f（CSP Bottleneck with 2 convolutions）
```yaml
- [-1, 3, C2f, [channels, shortcut]]
```
- YOLOv8 的核心模块
- `shortcut=True`：使用残差连接

### SPPF（Spatial Pyramid Pooling - Fast）
```yaml
- [-1, 1, SPPF, [channels, kernel_size]]
```
- 多尺度特征融合
- 快速版本的空间金字塔池化

### Concat（张量拼接）
```yaml
- [[layer1, layer2], 1, Concat, [dimension]]
```
- 在指定维度拼接多个特征图
- `dimension=1`：在通道维度拼接

### nn.Upsample（上采样）
```yaml
- [-1, 1, nn.Upsample, [size, scale_factor, mode]]
```
- `size=None`：使用 scale_factor
- `scale_factor=2`：放大2倍
- `mode="nearest"`：最近邻插值

### Detect（检测头）
```yaml
- [[P3, P4, P5], 1, Detect, [nc]]
```
- 多尺度检测输出
- `nc`：类别数量

---

## 实战示例

### 示例1：创建轻量级模型

```yaml
# 减少通道数和重复次数
nc: 80
scales:
  custom: [0.25, 0.25, 512]  # 更小的模型

backbone:
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 2, C2f, [64, True]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C2f, [128, True]]
  - [-1, 1, SPPF, [128, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2f, [128]]
  - [[8], 1, Detect, [nc]]
```

### 示例2：添加注意力机制

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, CBAM, [128]]        # 添加 CBAM 注意力
  - [-1, 1, Conv, [256, 3, 2]]
  # ... 其余层
```

### 示例3：自定义检测头

```yaml
head:
  # 只使用两个尺度检测（P3 和 P4）
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]         # P4

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]         # P3

  - [[14, 11], 1, Detect, [nc]]  # 双尺度检测
```

---

## 高级技巧

### 1. 调试配置文件

```python
from ultralytics import YOLO

# 加载并打印模型结构
model = YOLO('your_config.yaml')
print(model.model)

# 验证前向传播
import torch
x = torch.randn(1, 3, 640, 640)
output = model.model(x)
print(f"输出形状: {[o.shape for o in output]}")
```

### 2. 层索引计算

```python
# 负数索引：-1 表示上一层，-2 表示上上层
# 正数索引：从 0 开始计数

# 示例：
# layer 10: from=-1  -> 从 layer 9 获取输入
# layer 11: from=[-1, 6] -> 从 layer 10 和 layer 6 获取输入
```

### 3. 通道数计算

```python
# 实际通道数 = 配置通道数 × width_multiple
# 示例：yolov8n.yaml with Conv [64, 3, 2]
# 实际通道数 = 64 × 0.25 = 16
```

### 4. 重复次数计算

```python
# 实际重复次数 = max(round(配置重复次数 × depth_multiple), 1)
# 示例：yolov8n.yaml with C2f repeats=3
# 实际重复次数 = max(round(3 × 0.33), 1) = 1
```

## 附录

### A. 完整配置模板

```yaml
# Ultralytics YOLO Configuration Template

# 全局参数
nc: 80
scales:
  custom: [depth_multiple, width_multiple, max_channels]

# Backbone
backbone:
  # Stage 1: Input stem
  - [-1, 1, Conv, [64, 3, 2]]

  # Stage 2-5: Feature extraction
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]

  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]

  # Stage 6: SPPF
  - [-1, 1, SPPF, [1024, 5]]

# Head
head:
  # FPN
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]

  # PAN
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]

  # Detect
  - [[15, 18, 21], 1, Detect, [nc]]
```

### B. 参考资源

- [Ultralytics 官方文档](https://docs.ultralytics.com)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [YAML 语法规范](https://yaml.org/spec/)

---
