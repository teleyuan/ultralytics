"""
模型头部模块

本模块包含了YOLO系列模型的各种检测头部实现，包括:
- Detect: 基础目标检测头
- Segment: 实例分割头
- OBB: 旋转边界框检测头
- Pose: 关键点检测头
- Classify: 分类头
- RTDETRDecoder: RT-DETR解码器
- YOLOEDetect/YOLOESegment: 文本引导的检测和分割头
- v10Detect: YOLOv10检测头
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "Segment", "YOLOEDetect", "YOLOESegment", "v10Detect"


class Detect(nn.Module):
    """
    YOLO目标检测头部

    该类实现了YOLO模型中用于预测边界框和类别概率的检测头部。
    支持训练和推理模式，并具有可选的端到端检测能力。

    属性:
        dynamic (bool): 是否强制重建网格，用于动态输入尺寸
        export (bool): 导出模式标志，影响模型的输出格式
        format (str): 导出格式（如onnx、tflite等）
        end2end (bool): 端到端检测模式，启用one2one预测
        max_det (int): 每张图像的最大检测数量
        shape (tuple): 输入形状，用于缓存避免重复计算
        anchors (torch.Tensor): 锚点坐标，从特征图生成
        strides (torch.Tensor): 特征图的步长（相对于原始图像）
        legacy (bool): 向后兼容标志，用于v3/v5/v8/v9模型
        xyxy (bool): 输出格式标志，True为xyxy格式，False为xywh格式
        nc (int): 类别数量
        nl (int): 检测层数量（通常为3层：大、中、小目标）
        reg_max (int): DFL（Distribution Focal Loss）通道数
        no (int): 每个锚点的输出数量（类别数 + 边界框参数数）
        stride (torch.Tensor): 在构建时计算的步长
        cv2 (nn.ModuleList): 用于边界框回归的卷积层列表
        cv3 (nn.ModuleList): 用于分类的卷积层列表
        dfl (nn.Module): 分布焦点损失层，用于边界框精细化
        one2one_cv2 (nn.ModuleList): 用于one2one边界框回归的卷积层
        one2one_cv3 (nn.ModuleList): 用于one2one分类的卷积层

    方法:
        forward: 执行前向传播并返回预测结果
        forward_end2end: 执行端到端检测的前向传播
        bias_init: 初始化检测头部的偏置
        decode_bboxes: 从预测中解码边界框
        postprocess: 后处理模型预测结果

    使用示例:
        创建一个80类的检测头
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # 强制重建网格
    export = False  # 导出模式
    format = None  # 导出格式
    end2end = False  # 端到端模式
    max_det = 300  # 最大检测数
    shape = None  # 输入形状缓存
    anchors = torch.empty(0)  # 锚点初始化
    strides = torch.empty(0)  # 步长初始化
    legacy = False  # v3/v5/v8/v9模型的向后兼容性
    xyxy = False  # 输出格式: xyxy或xywh

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        初始化YOLO检测层

        Args:
            nc (int): 类别数量，默认80（COCO数据集）
            ch (tuple): 来自骨干网络特征图的通道大小元组，通常是3个不同尺度的特征图
        """
        super().__init__()
        self.nc = nc  # 类别数量
        self.nl = len(ch)  # 检测层数量，通常为3（对应不同尺度）
        self.reg_max = 16  # DFL通道数（ch[0] // 16用于缩放n/s/m/l/x模型的4/8/12/16/20）
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数量（类别数 + 4*reg_max个边界框参数）
        self.stride = torch.zeros(self.nl)  # 在构建时计算的步长

        # 计算通道数：c2用于边界框回归，c3用于分类
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        # 边界框回归头：两个3x3卷积 + 一个1x1卷积输出
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        # 分类头：根据是否为legacy模式使用不同的架构
        self.cv3 = (
            # legacy模式：使用标准卷积
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            # 新版本：使用深度可分离卷积(DWConv)提高效率
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),  # 深度可分离卷积 + 1x1卷积
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),  # 最终输出层
                )
                for x in ch
            )
        )

        # 分布焦点损失层，用于将分布转换为边界框坐标
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 端到端模式：复制一套one2one预测头
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """
        前向传播：拼接并返回预测的边界框和类别概率

        Args:
            x: 来自骨干网络的特征图列表，通常包含3个不同尺度的特征图

        Returns:
            训练模式: 返回原始预测列表
            推理模式: 返回处理后的预测结果（导出模式）或 (处理结果, 原始预测) 元组
        """
        if self.end2end:
            return self.forward_end2end(x)

        # 对每个检测层进行预测
        for i in range(self.nl):
            # 拼接边界框预测和分类预测
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:  # 训练路径：直接返回原始预测
            return x

        # 推理路径：解码预测结果
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: list[torch.Tensor]) -> dict | tuple:
        """
        执行端到端检测模式的前向传播（v10Detect模块）

        Args:
            x (list[torch.Tensor]): 来自不同层级的输入特征图

        Returns:
            outputs (dict | tuple):
                训练模式: 返回包含one2many和one2one输出的字典
                推理模式: 返回处理后的检测结果或 (检测结果, 原始输出) 元组
        """
        # 分离特征图用于one2one预测（避免梯度传播）
        x_detach = [xi.detach() for xi in x]

        # one2one预测分支：每个目标只匹配一个预测
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]

        # one2many预测分支：每个目标可以匹配多个预测（常规训练方式）
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:  # 训练路径：返回两个分支的预测
            return {"one2many": x, "one2one": one2one}

        # 推理路径：只使用one2one分支进行预测
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        基于多层级特征图解码预测的边界框和类别概率

        Args:
            x (list[torch.Tensor]): 来自不同检测层的特征图列表

        Returns:
            (torch.Tensor): 拼接后的解码边界框和类别概率张量
        """
        # 推理路径
        shape = x[0].shape  # BCHW格式（批次、通道、高度、宽度）
        # 将所有层的预测拼接：将每个特征图reshape为 (batch, no, -1) 然后在维度2上拼接
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # 如果是动态模式或形状改变，重新生成锚点和步长
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 分离边界框预测和类别预测
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 解码边界框：DFL将分布转换为距离，然后转换为实际坐标
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # 拼接解码后的边界框和sigmoid激活后的类别概率
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """
        初始化检测头的偏置
        警告：需要步长(stride)已经可用

        该方法初始化边界框和分类头的偏置，以提高训练初期的稳定性
        """
        m = self  # 检测模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 标称类别频率

        # 初始化常规检测头
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # 边界框偏置设为1.0
            # 分类偏置：基于类别数和特征图步长计算（假设每640像素图像有0.01个目标）
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

        # 如果是端到端模式，也初始化one2one头
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0  # 边界框偏置
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 分类偏置

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """
        从预测中解码边界框

        Args:
            bboxes: 预测的边界框张量
            anchors: 锚点坐标
            xywh: 是否返回xywh格式（否则返回xyxy格式）

        Returns:
            解码后的边界框
        """
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """
        后处理YOLO模型预测结果

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc)
                                 最后一维格式为 [x, y, w, h, class_probs]
            max_det (int): 每张图像的最大检测数量
            nc (int, optional): 类别数量

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6)
                           最后一维格式为 [x, y, w, h, max_class_prob, class_index]
        """
        batch_size, anchors, _ = preds.shape  # 例如 shape(16, 8400, 84)

        # 分离边界框和分数
        boxes, scores = preds.split([4, nc], dim=-1)

        # 获取每个锚点的最高分数，并选择top-k个
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))

        # 在所有类别中选择top-k个最高分数
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # 批次索引

        # 拼接：边界框 + 最大类别概率 + 类别索引
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """
    YOLO实例分割头部

    该类扩展了Detect检测头，增加了用于实例分割任务的掩码预测能力。
    采用原型网络(Proto Network)方法，通过原型和系数的线性组合生成实例掩码。

    属性:
        nm (int): 掩码数量（掩码系数的数量）
        npr (int): 原型数量（原型特征图的通道数）
        proto (Proto): 原型生成模块，生成掩码原型
        cv4 (nn.ModuleList): 用于预测掩码系数的卷积层

    方法:
        forward: 返回模型输出和掩码系数

    使用示例:
        创建一个分割头
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: tuple = ()):
        """
        初始化YOLO分割模型

        Args:
            nc (int): 类别数量
            nm (int): 掩码系数数量，通常为32
            npr (int): 原型数量（原型特征图通道数），通常为256
            ch (tuple): 来自骨干网络特征图的通道大小元组
        """
        super().__init__(nc, ch)
        self.nm = nm  # 掩码系数数量
        self.npr = npr  # 原型数量
        # 原型生成模块：从最细粒度特征图生成掩码原型
        self.proto = Proto(ch[0], self.npr, self.nm)

        # 掩码系数预测头
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor]:
        """
        前向传播：返回检测结果和掩码相关信息

        Args:
            x: 来自骨干网络的特征图列表

        Returns:
            训练模式: (检测预测, 掩码系数, 掩码原型)
            推理模式: (拼接的检测和掩码系数, 掩码原型) 或 ((检测结果, 原始预测), 掩码系数, 掩码原型)
        """
        p = self.proto(x[0])  # 从最细粒度特征生成掩码原型 (bs, nm, h, w)
        bs = p.shape[0]  # 批次大小

        # 预测掩码系数：每个检测框对应nm个系数
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)

        # 调用父类的检测前向传播
        x = Detect.forward(self, x)

        if self.training:
            return x, mc, p

        # 推理模式：拼接检测结果和掩码系数
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """
    YOLO旋转边界框(OBB)检测头

    该类扩展了Detect检测头，增加了带旋转角度的定向边界框预测能力。
    适用于需要精确检测目标方向的场景，如航拍图像、文本检测等。

    属性:
        ne (int): 额外参数数量（通常为1，表示旋转角度）
        cv4 (nn.ModuleList): 用于角度预测的卷积层
        angle (torch.Tensor): 预测的旋转角度

    方法:
        forward: 拼接并返回预测的边界框和类别概率
        decode_bboxes: 解码旋转边界框

    使用示例:
        创建一个OBB检测头
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, ch: tuple = ()):
        """
        初始化OBB检测头

        Args:
            nc (int): 类别数量
            ne (int): 额外参数数量（旋转角度参数）
            ch (tuple): 来自骨干网络特征图的通道大小元组
        """
        super().__init__(nc, ch)
        self.ne = ne  # 额外参数数量（角度）

        # 角度预测头
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """
        前向传播：拼接并返回预测的旋转边界框和类别概率

        Args:
            x: 来自骨干网络的特征图列表

        Returns:
            训练模式: (检测预测, 角度)
            推理模式: 拼接的预测结果 或 (拼接结果, 原始预测)
        """
        bs = x[0].shape[0]  # 批次大小

        # 预测旋转角度（theta）的logits
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)

        # 注意：将angle设为属性，以便decode_bboxes方法可以使用
        # 将角度归一化到 [-pi/4, 3pi/4] 范围（覆盖180度）
        angle = (angle.sigmoid() - 0.25) * math.pi
        # 备选方案：angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]

        if not self.training:
            self.angle = angle

        # 调用父类的检测前向传播
        x = Detect.forward(self, x)

        if self.training:
            return x, angle

        # 推理模式：拼接检测结果和角度
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        解码旋转边界框

        Args:
            bboxes: 预测的边界框张量
            anchors: 锚点坐标

        Returns:
            解码后的旋转边界框（包含角度信息）
        """
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """
    YOLO姿态估计头部

    该类扩展了Detect检测头，增加了用于姿态估计任务的关键点预测能力。
    可以同时检测目标并预测其身体关键点位置（如人体的17个关键点）。

    属性:
        kpt_shape (tuple): 关键点形状，(关键点数量, 维度数)
                          维度为2表示(x,y)，维度为3表示(x,y,visible)
        nk (int): 关键点总数值（关键点数量 × 维度数）
        cv4 (nn.ModuleList): 用于关键点预测的卷积层

    方法:
        forward: 执行前向传播并返回预测结果
        kpts_decode: 从预测中解码关键点

    使用示例:
        创建一个姿态检测头（COCO人体17个关键点）
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), ch: tuple = ()):
        """
        初始化YOLO姿态估计网络

        Args:
            nc (int): 类别数量
            kpt_shape (tuple): 关键点形状 (关键点数量, 维度)
                             例如 (17, 3) 表示17个关键点，每个3个值(x, y, visible)
            ch (tuple): 来自骨干网络特征图的通道大小元组
        """
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # 关键点形状
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点总参数数量

        # 关键点预测头
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """
        前向传播：执行检测和关键点预测

        Args:
            x: 来自骨干网络的特征图列表

        Returns:
            训练模式: (检测预测, 关键点预测)
            推理模式: 拼接的预测结果 或 (拼接结果, 原始预测)
        """
        bs = x[0].shape[0]  # 批次大小

        # 预测关键点：(bs, nk, h*w) 例如 (bs, 17*3, h*w)
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)

        # 调用父类的检测前向传播
        x = Detect.forward(self, x)

        if self.training:
            return x, kpt

        # 解码关键点坐标
        pred_kpt = self.kpts_decode(bs, kpt)

        # 推理模式：拼接检测结果和关键点
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        """
        从预测中解码关键点坐标

        Args:
            bs: 批次大小
            kpts: 预测的关键点张量

        Returns:
            解码后的关键点坐标（相对于原始图像）
        """
        ndim = self.kpt_shape[1]  # 每个关键点的维度数

        if self.export:
            # NCNN导出修复
            y = kpts.view(bs, *self.kpt_shape, -1)
            # 解码xy坐标：从归一化坐标转换为实际像素坐标
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                # 添加可见性sigmoid激活
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                # 对可见性进行sigmoid激活
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()  # 就地操作
                else:  # 解决Apple macOS14 MPS bug
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()

            # 解码x坐标
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            # 解码y坐标
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """
    YOLO分类头部

    该类实现了图像分类头，将特征图转换为类别预测。
    输入形状 x(b,c1,h,w) -> 输出形状 x(b,c2)

    属性:
        export (bool): 导出模式标志
        conv (Conv): 特征转换卷积层
        pool (nn.AdaptiveAvgPool2d): 全局平均池化层，将空间维度压缩为1x1
        drop (nn.Dropout): Dropout层，用于正则化防止过拟合
        linear (nn.Linear): 线性层，输出最终类别预测

    方法:
        forward: 对输入图像数据执行前向传播

    使用示例:
        创建一个1000类的分类头（如ImageNet）
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # 导出模式

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """
        初始化YOLO分类头

        将输入张量从 (b,c1,h,w) 转换为 (b,c2) 形状

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出类别数
            k (int, optional): 卷积核大小
            s (int, optional): 步长
            p (int, optional): 填充
            g (int, optional): 分组数
        """
        super().__init__()
        c_ = 1280  # EfficientNet-B0的中间通道数
        self.conv = Conv(c1, c_, k, s, p, g)  # 特征转换
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化: (b,c_,h,w) -> (b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)  # Dropout正则化
        self.linear = nn.Linear(c_, c2)  # 全连接层: (b,c_) -> (b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """
        执行分类头的前向传播

        Args:
            x: 输入特征图，可以是单个张量或张量列表

        Returns:
            训练模式: 返回原始logits
            推理模式: 返回softmax概率 或 (概率, logits) 元组
        """
        # 如果输入是列表，先在通道维度拼接
        if isinstance(x, list):
            x = torch.cat(x, 1)

        # 前向传播流程: Conv -> Pool -> Flatten -> Dropout -> Linear
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

        if self.training:
            return x  # 训练时返回logits用于计算损失

        # 推理时应用softmax得到类别概率
        y = x.softmax(1)
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        """Fuse text features with model weights for efficient inference."""
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()

        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward_lrpc(self, x: list[torch.Tensor], return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x: list[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for detection heads."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self, nc: int = 80, nm: int = 32, npr: int = 256, embed: int = 512, with_bn: bool = False, ch: tuple = ()
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> tuple | torch.Tensor:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mc, p

        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    """
    实时可变形Transformer解码器（RT-DETR Decoder）

    该解码器模块结合了Transformer架构和可变形卷积来预测图像中目标的边界框和类别标签。
    它整合了多层特征，并通过一系列Transformer解码器层输出最终预测结果。

    RT-DETR是一种端到端的实时目标检测器，无需NMS后处理。

    属性:
        export (bool): 导出模式标志
        hidden_dim (int): 隐藏层维度
        nhead (int): 多头注意力的头数
        nl (int): 特征层级数量
        nc (int): 类别数量
        num_queries (int): 查询点数量（检测数量）
        num_decoder_layers (int): 解码器层数
        input_proj (nn.ModuleList): 骨干特征的输入投影层
        decoder (DeformableTransformerDecoder): 可变形Transformer解码器模块
        denoising_class_embed (nn.Embedding): 用于去噪的类别嵌入
        num_denoising (int): 去噪查询数量
        label_noise_ratio (float): 训练时的标签噪声比例
        box_noise_scale (float): 训练时的边界框噪声尺度
        learnt_init_query (bool): 是否学习初始查询嵌入
        tgt_embed (nn.Embedding): 查询的目标嵌入
        query_pos_head (MLP): 查询位置头
        enc_output (nn.Sequential): 编码器输出层
        enc_score_head (nn.Linear): 编码器分数预测头
        enc_bbox_head (MLP): 编码器边界框预测头
        dec_score_head (nn.ModuleList): 解码器分数预测头
        dec_bbox_head (nn.ModuleList): 解码器边界框预测头

    方法:
        forward: 执行前向传播并返回边界框和分类分数

    使用示例:
        创建一个RT-DETR解码器
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim 隐藏维度
        nq: int = 300,  # num queries 查询数量
        ndp: int = 4,  # num decoder points 解码器采样点数
        nh: int = 8,  # num head 注意力头数
        ndl: int = 6,  # num decoder layers 解码器层数
        d_ffn: int = 1024,  # dim of feedforward 前馈网络维度
        dropout: float = 0.0,  # dropout比例
        act: nn.Module = nn.ReLU(),  # 激活函数
        eval_idx: int = -1,  # 评估索引
        # 训练参数
        nd: int = 100,  # num denoising 去噪数量
        label_noise_ratio: float = 0.5,  # 标签噪声比例
        box_noise_scale: float = 1.0,  # 边界框噪声尺度
        learnt_init_query: bool = False,  # 是否学习初始查询
    ):
        """
        使用给定参数初始化RT-DETR解码器模块

        Args:
            nc (int): 类别数量
            ch (tuple): 骨干网络特征图的通道数
            hd (int): 隐藏层维度（通常为256）
            nq (int): 查询点数量（即最大检测数量，通常为300）
            ndp (int): 解码器采样点数量（可变形注意力的采样点）
            nh (int): 多头注意力的头数
            ndl (int): 解码器层数（通常为6层）
            d_ffn (int): 前馈网络的维度
            dropout (float): Dropout比例
            act (nn.Module): 激活函数
            eval_idx (int): 评估索引（-1表示使用最后一层）
            nd (int): 去噪查询数量（用于训练加速）
            label_noise_ratio (float): 训练时的标签噪声比例
            box_noise_scale (float): 训练时的边界框噪声尺度
            learnt_init_query (bool): 是否学习初始查询嵌入
        """
        super().__init__()
        self.hidden_dim = hd  # 隐藏维度
        self.nhead = nh  # 注意力头数
        self.nl = len(ch)  # 特征层级数量
        self.nc = nc  # 类别数量
        self.num_queries = nq  # 查询数量
        self.num_decoder_layers = ndl  # 解码器层数

        # 骨干特征投影层：将不同通道的特征统一投影到隐藏维度
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 注意：简化版本但与.pt权重不一致
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer模块：可变形注意力解码器
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # 去噪训练部分：通过添加噪声查询加速训练收敛
        self.denoising_class_embed = nn.Embedding(nc, hd)  # 类别嵌入用于去噪
        self.num_denoising = nd  # 去噪查询数量
        self.label_noise_ratio = label_noise_ratio  # 标签噪声比例
        self.box_noise_scale = box_noise_scale  # 边界框噪声尺度

        # 解码器嵌入
        self.learnt_init_query = learnt_init_query  # 是否学习初始查询
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)  # 可学习的查询嵌入
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)  # 查询位置编码头

        # 编码器输出头：用于选择top-k特征作为初始查询
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)  # 编码器分类头
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)  # 编码器边界框头

        # 解码器输出头：每一层都有独立的预测头
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()  # 初始化参数

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """
        执行模块的前向传播，返回边界框和分类分数

        Args:
            x (list[torch.Tensor]): 来自骨干网络的特征图列表
            batch (dict, optional): 训练时的批次信息

        Returns:
            outputs (tuple | torch.Tensor):
                训练模式: 返回包含边界框、分数和其他元数据的元组
                推理模式: 返回形状为 (bs, 300, 4+nc) 的张量，包含边界框和类别分数
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # 1. 输入投影和嵌入：将多尺度特征投影到统一维度
        feats, shapes = self._get_encoder_input(x)

        # 2. 准备去噪训练：生成带噪声的查询以加速训练
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        # 3. 获取解码器输入：从编码器特征中选择top-k作为初始查询
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 4. 解码器：通过多层可变形注意力迭代优化查询
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        # 5. 返回结果
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x

        # 推理模式：拼接边界框和分类分数 (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为给定形状生成锚点边界框并验证它们

        该方法生成均匀分布的锚点网格，作为RT-DETR的初始参考点。

        Args:
            shapes (list): 特征图形状列表 [[h1,w1], [h2,w2], ...]
            grid_size (float, optional): 网格单元的基础大小（归一化值）
            dtype (torch.dtype, optional): 张量数据类型
            device (str, optional): 创建张量的设备
            eps (float, optional): 数值稳定性的小值

        Returns:
            anchors (torch.Tensor): 生成的锚点框 (1, h*w*nl, 4)，格式为logit空间的(cx, cy, w, h)
            valid_mask (torch.Tensor): 锚点的有效掩码 (1, h*w*nl, 1)
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            # 创建网格坐标
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            # 生成网格：根据PyTorch版本使用不同的索引方式
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2) 堆叠为xy坐标

            # 归一化坐标到[0,1]范围
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2) 中心点坐标

            # 设置锚点大小：不同层级使用不同的基础尺寸
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)

            # 拼接中心坐标和宽高：(cx, cy, w, h)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        # 合并所有层级的锚点
        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)

        # 创建有效掩码：过滤掉太靠近边界的锚点
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # (1, h*w*nl, 1)

        # 转换到logit空间：将[0,1]范围转换为(-inf, inf)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))  # 无效锚点设为inf
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """
        处理并返回编码器输入

        通过投影层统一多尺度特征的维度，并将其展平拼接。

        Args:
            x (list[torch.Tensor]): 来自骨干网络的特征图列表

        Returns:
            feats (torch.Tensor): 处理后的特征 (b, h*w*nl, c)
            shapes (list): 特征图形状列表 [[h1,w1], [h2,w2], ...]
        """
        # 投影特征：将不同通道统一到隐藏维度
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]

        # 获取编码器输入
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # 空间展平：[b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # 记录形状信息
            shapes.append([h, w])

        # 拼接所有层级的特征：[b, h*w*nl, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成并准备解码器所需的输入

        从编码器特征中选择top-k个最有可能包含目标的位置作为初始查询。

        Args:
            feats (torch.Tensor): 来自编码器的处理特征
            shapes (list): 特征图形状列表
            dn_embed (torch.Tensor, optional): 去噪嵌入
            dn_bbox (torch.Tensor, optional): 去噪边界框

        Returns:
            embeddings (torch.Tensor): 解码器的查询嵌入
            refer_bbox (torch.Tensor): 参考边界框
            enc_bboxes (torch.Tensor): 编码的边界框
            enc_scores (torch.Tensor): 编码的分数
        """
        bs = feats.shape[0]
        # 如果是动态模式或形状改变，重新生成锚点
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # 准备解码器输入
        # 1. 通过编码器输出层处理特征（只处理有效区域）
        features = self.enc_output(self.valid_mask * feats)  # (bs, h*w, 256)

        # 2. 预测每个位置的类别分数
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # 3. 查询选择：选择top-k个最有可能包含目标的位置
        # 对每个位置取最大类别分数，然后选择top-k
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)  # (bs*nq,)
        # 创建批次索引
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # 4. 提取top-k位置的特征和锚点
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # (bs, nq, 256)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)  # (bs, nq, 4)

        # 5. 生成参考边界框：动态锚点 + 静态内容
        # 预测相对于锚点的偏移量
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        # 6. 编码边界框（sigmoid到[0,1]范围）
        enc_bboxes = refer_bbox.sigmoid()
        # 如果有去噪边界框，拼接到前面
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        # 记录编码器预测的分数
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # 7. 生成查询嵌入
        # 如果学习初始查询，使用可学习的嵌入；否则使用top-k特征
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features

        # 8. 训练时分离梯度
        if self.training:
            refer_bbox = refer_bbox.detach()  # 停止梯度传播
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        # 9. 如果有去噪嵌入，拼接到前面
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """
        初始化或重置模型各组件的参数

        使用预定义的权重和偏置初始化策略。
        """
        # 分类和边界框头初始化
        # 计算分类偏置：基于先验概率（假设每个位置有1%概率包含目标）
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc

        # 注意：linear_init中的权重初始化在自定义数据集训练时可能导致NaN
        # linear_init(self.enc_score_head)

        # 初始化编码器头
        constant_(self.enc_score_head.bias, bias_cls)  # 分类偏置
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)  # 边界框最后一层权重设为0
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)  # 边界框最后一层偏置设为0

        # 初始化解码器头（每一层）
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)  # 分类偏置
            constant_(reg_.layers[-1].weight, 0.0)  # 边界框权重
            constant_(reg_.layers[-1].bias, 0.0)  # 边界框偏置

        # Xavier初始化
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)

        # 如果使用可学习的初始查询，初始化查询嵌入
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)

        # 初始化查询位置头
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)

        # 初始化输入投影层
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    YOLOv10检测头
    论文链接: https://arxiv.org/pdf/2405.14458

    该类实现了YOLOv10检测头，采用双重分配训练和一致的双重预测，
    以提高效率和性能。主要创新是同时使用one2many和one2one两种匹配策略。

    属性:
        end2end (bool): 端到端检测模式（默认为True）
        max_det (int): 最大检测数量
        cv3 (nn.ModuleList): 轻量级分类头层
        one2one_cv3 (nn.ModuleList): One2one分类头层

    方法:
        __init__: 使用指定的类别数和输入通道初始化v10Detect对象
        forward: 执行v10Detect模块的前向传播
        bias_init: 初始化Detect模块的偏置
        fuse: 移除one2many头以优化推理

    使用示例:
        创建一个v10Detect头
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True  # 默认启用端到端模式

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        初始化v10Detect对象

        Args:
            nc (int): 类别数量
            ch (tuple): 来自骨干网络特征图的通道大小元组
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # 计算分类头通道数

        # 轻量级分类头：使用分组卷积减少参数量
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),  # 深度可分离卷积
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),  # 深度可分离卷积
                nn.Conv2d(c3, self.nc, 1),  # 1x1卷积输出类别
            )
            for x in ch
        )
        # One2one分类头：用于端到端训练
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """
        移除one2many头以优化推理

        在推理时只需要one2one分支，移除one2many分支可以加速推理。
        """
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)
