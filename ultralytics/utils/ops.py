"""
Ultralytics 操作函数模块

该模块包含了 YOLO 模型中使用的核心数学和图像处理操作函数。
这些函数涵盖了边界框操作、坐标转换、非极大值抑制 (NMS)、图像缩放等关键功能。

主要功能:
    - 边界框操作: 格式转换、缩放、裁剪、IoU 计算等
    - 坐标变换: xyxy/xywh/ltwh 等格式相互转换
    - 非极大值抑制: NMS 和 Soft-NMS 实现
    - 图像处理: 仿射变换、letterbox、scale_image 等
    - 分割操作: mask 处理、多边形转换、crop 等
    - 性能分析: Profile 类用于代码执行计时

核心类:
    - Profile: 代码性能分析和计时工具

核心函数:
    - xyxy2xywh/xywh2xyxy: 边界框格式转换
    - scale_boxes/scale_coords: 边界框坐标缩放
    - clip_boxes/clip_coords: 边界框坐标裁剪
    - non_max_suppression: 非极大值抑制
    - box_iou: 边界框 IoU 计算
    - masks2segments: mask 转分割多边形

算法说明:
    - NMS: 移除重叠度高的冗余检测框
    - IoU (Intersection over Union): 计算框重叠度
    - Affine Transform: 仿射变换用于数据增强
"""

from __future__ import annotations  # 支持 Python 3.7+ 的类型注解

# 标准库导入
import contextlib  # 上下文管理器工具
import math  # 数学函数
import re  # 正则表达式
import time  # 时间函数

# 第三方库导入
import cv2  # OpenCV - 图像处理
import numpy as np  # NumPy - 数值计算
import torch  # PyTorch - 深度学习框架
import torch.nn.functional as F  # PyTorch 函数式 API

# Ultralytics 内部导入
from ultralytics.utils import NOT_MACOS14  # macOS 14 版本检测标志


class Profile(contextlib.ContextDecorator):
    """
    Ultralytics 性能分析类 - 用于代码执行计时
    Ultralytics Profile class for timing code execution.

    可作为装饰器 @Profile() 或上下文管理器 'with Profile():' 使用。
    提供精确的计时测量，支持 GPU 操作的 CUDA 同步。

    Use as a decorator with @Profile() or as a context manager with 'with Profile():'. Provides accurate timing
    measurements with CUDA synchronization support for GPU operations.

    属性:
        t (float): 累积时间 (秒)
        device (torch.device): 用于模型推理的设备
        cuda (bool): 是否使用 CUDA 进行时间同步

    Attributes:
        t (float): Accumulated time in seconds.
        device (torch.device): Device used for model inference.
        cuda (bool): Whether CUDA is being used for timing synchronization.

    示例:
        作为上下文管理器使用:
        >>> with Profile(device=device) as dt:
        ...     pass  # 耗时操作
        >>> print(dt)  # 打印 "Elapsed time is 9.5367431640625e-07 s"

        作为装饰器使用:
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)

    Examples:
        Use as a context manager to time code execution
        >>> with Profile(device=device) as dt:
        ...     pass  # slow operation here
        >>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

        Use as a decorator to time function execution
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """
        初始化 Profile 类
        Initialize the Profile class.

        Args:
            t (float): 初始累积时间 (秒)
            device (torch.device, optional): 用于模型推理的设备，以启用 CUDA 同步
        """
        self.t = t  # 累积时间
        self.device = device  # 设备
        self.cuda = bool(device and str(device).startswith("cuda"))  # 是否为 CUDA 设备

    def __enter__(self):
        """
        进入上下文管理器时开始计时
        Start timing.
        """
        self.start = self.time()  # 记录开始时间
        return self

    def __exit__(self, type, value, traceback):
        """
        退出上下文管理器时停止计时
        Stop timing.
        """
        self.dt = self.time() - self.start  # 计算时间增量 (delta-time)
        self.t += self.dt  # 累积到总时间

    def __str__(self):
        """
        返回表示累积经过时间的人类可读字符串
        Return a human-readable string representing the accumulated elapsed time.
        """
        return f"Elapsed time is {self.t} s"

    def time(self):
        """
        获取当前时间，如适用则进行 CUDA 同步
        Get current time with CUDA synchronization if applicable.

        Note:
            CUDA 同步确保 GPU 操作完成后再记录时间，避免异步执行导致的计时不准确
        """
        if self.cuda:
            torch.cuda.synchronize(self.device)  # 等待 CUDA 操作完成
        return time.perf_counter()  # 返回高精度计时器值


def segment2box(segment, width: int = 640, height: int = 640):
    """
    将分割坐标转换为边界框坐标
    Convert segment coordinates to bounding box coordinates.

    通过找到最小和最大 x、y 坐标将单个分割标签转换为边界框标签。
    应用图像内约束，必要时裁剪坐标。

    Converts a single segment label to a box label by finding the minimum and maximum x and y coordinates. Applies
    inside-image constraint and clips coordinates when necessary.

    Args:
        segment (torch.Tensor): 分割坐标，格式为 (N, 2)，其中 N 是点的数量
        width (int): 图像宽度 (像素)
        height (int): 图像高度 (像素)

    Returns:
        (np.ndarray): xyxy 格式的边界框坐标 [x1, y1, x2, y2]

    Examples:
        >>> segment = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
        >>> box = segment2box(segment, 640, 640)
        >>> print(box)  # [10, 10, 20, 20]
    """
    x, y = segment.T  # 转置分割坐标，分离 x 和 y
    # 如果 4 条边中有 3 条在图像外，则裁剪坐标
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)  # 裁剪 x 坐标到 [0, width]
        y = y.clip(0, height)  # 裁剪 y 坐标到 [0, height]
    # 只保留图像内的点
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    # 返回边界框 [x_min, y_min, x_max, y_max]，如果没有有效点则返回零数组
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy 格式


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    将边界框从一个图像形状重新缩放到另一个图像形状
    Rescale bounding boxes from one image shape to another.

    将边界框从 img1_shape 重新缩放到 img0_shape，考虑填充和纵横比变化。
    支持 xyxy 和 xywh 两种边界框格式。

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Supports
    both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): 源图像形状 (高度, 宽度)
        boxes (torch.Tensor): 要重新缩放的边界框，格式为 (N, 4)
        img0_shape (tuple): 目标图像形状 (高度, 宽度)
        ratio_pad (tuple, optional): 缩放的 (比例, 填充) 元组。如果为 None，则从图像形状计算
        padding (bool): 边界框是否基于带填充的 YOLO 风格增强图像
        xywh (bool): 边界框格式是 xywh (True) 还是 xyxy (False)

    Returns:
        (torch.Tensor): 重新缩放后的边界框，格式与输入相同

    Examples:
        >>> boxes = torch.tensor([[50, 50, 100, 100]])
        >>> img1_shape = (640, 640)
        >>> img0_shape = (480, 640)
        >>> scaled_boxes = scale_boxes(img1_shape, boxes, img0_shape)
    """
    if ratio_pad is None:  # 从 img0_shape 计算
        # 计算缩放增益 (保持纵横比)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        # 计算填充 (居中图像)
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]  # 使用提供的增益
        pad_x, pad_y = ratio_pad[1]  # 使用提供的填充

    if padding:
        # 移除填充
        boxes[..., 0] -= pad_x  # x 填充
        boxes[..., 1] -= pad_y  # y 填充
        if not xywh:
            boxes[..., 2] -= pad_x  # x 填充 (仅对 xyxy 格式)
            boxes[..., 3] -= pad_y  # y 填充 (仅对 xyxy 格式)
    # 除以增益进行缩放
    boxes[..., :4] /= gain
    # 如果是 xyxy 格式，则裁剪到图像边界
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """
    返回能被给定除数整除的最接近的数
    Return the nearest number that is divisible by the given divisor.

    Args:
        x (int): 要使其可整除的数
        divisor (int | torch.Tensor): 除数

    Returns:
        (int): 能被除数整除的最接近的数

    Examples:
        >>> make_divisible(37, 8)
        40
        >>> make_divisible(42, 8)
        48

    Note:
        此函数通常用于确保网络层的通道数是某个值的倍数，
        以便更好地利用硬件加速 (如 Tensor Cores 通常要求 8 的倍数)
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # 转换为整数
    # 向上取整到最接近的除数倍数
    return math.ceil(x / divisor) * divisor


def clip_boxes(boxes, shape):
    """Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor):  # faster individually
        if NOT_MACOS14:
            boxes[..., 0].clamp_(0, w)  # x1
            boxes[..., 1].clamp_(0, h)  # y1
            boxes[..., 2].clamp_(0, w)  # x2
            boxes[..., 3].clamp_(0, h)  # y2
        else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """Clip line coordinates to image boundaries.

    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14:
            coords[..., 0].clamp_(0, w)  # x
            coords[..., 1].clamp_(0, h)  # y
        else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:  # np.array
        coords[..., 0] = coords[..., 0].clip(0, w)  # x
        coords[..., 1] = coords[..., 1].clip(0, h)  # y
    return coords


def xyxy2xywh(x):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, w, h) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        padw (int): Padding width in pixels.
        padh (int): Padding height in pixels.

    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where x1,y1 is
            the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    y[..., 0] = w * (xc - half_w) + padw  # top left x
    y[..., 1] = h * (yc - half_h) + padh  # top left y
    y[..., 2] = w * (xc + half_w) + padw  # bottom right x
    y[..., 3] = h * (yc + half_h) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        clip (bool): Whether to clip boxes to image boundaries.
        eps (float): Minimum value for box width and height.

    Returns:
        (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, width, height) format.
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = ((x1 + x2) / 2) / w  # x center
    y[..., 1] = ((y1 + y2) / 2) / h  # y center
    y[..., 2] = (x2 - x1) / w  # width
    y[..., 3] = (y2 - y1) / h  # height
    return y


def xywh2ltwh(x):
    """Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h] format.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xyxy format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    """Convert bounding boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxyxyxy2xywhr(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

    Args:
        x (np.ndarray | torch.Tensor): Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.

    Returns:
        (np.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation
            values are in radians from 0 to pi/2.
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.

    Args:
        x (np.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation
            values should be in radians from 0 to pi/2.

    Returns:
        (np.ndarray | torch.Tensor): Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyxy format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # x2
    y[..., 3] = x[..., 3] + x[..., 1]  # y2
    return y


def segments2boxes(segments):
    """Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): List of segments where each segment is a list of points, each point is [x, y] coordinates.

    Returns:
        (np.ndarray): Bounding box coordinates in xywh format.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n: int = 1000):
    """Resample segments to n points each using linear interpolation.

    Args:
        segments (list): List of (N, 2) arrays where N is the number of points in each segment.
        n (int): Number of points to resample each segment to.

    Returns:
        (list): Resampled segments with n points each.
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Crop masks to bounding box regions.

    Args:
        masks (torch.Tensor): Masks with shape (N, H, W).
        boxes (torch.Tensor): Bounding box coordinates with shape (N, 4) in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    n, h, w = masks.shape
    if n < 50 and not masks.is_cuda:  # faster for fewer masks (predict)
        for i, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:  # faster for more masks (val)
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """Apply masks to bounding boxes using mask head output.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # NHW

    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device)

    masks = crop_mask(masks, boxes=bboxes * ratios)  # NHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]  # NHW
    return masks.gt_(0.0).byte()


def process_mask_native(protos, masks_in, bboxes, shape):
    """Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (N, H, W).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # NHW
    masks = crop_mask(masks, bboxes)  # NHW
    return masks.gt_(0.0).byte()


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """Rescale segment masks to target shape.

    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple[int, int]): Target height and width as (height, width).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks

    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_h / im0_h, im1_w / im0_w)  # gain  = old / new
        pad_w, pad_h = (im1_w - im0_w * gain), (im1_h - im0_h * gain)  # wh padding
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(masks[..., top:bottom, left:right].float(), shape, mode="bilinear")  # NCHW masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """Rescale segment coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Source image shape as HWC or HW (supports both).
        coords (torch.Tensor): Coordinates to scale with shape (N, 2).
        img0_shape (tuple): Image 0 shape as HWC or HW (supports both).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).
        normalize (bool): Whether to normalize coordinates to range [0, 1].
        padding (bool): Whether coordinates are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Scaled coordinates.
    """
    img0_h, img0_w = img0_shape[:2]  # supports both HWC or HW shapes
    if ratio_pad is None:  # calculate from img0_shape
        img1_h, img1_w = img1_shape[:2]  # supports both HWC or HW shapes
        gain = min(img1_h / img0_h, img1_w / img0_w)  # gain  = old / new
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w  # width
        coords[..., 1] /= img0_h  # height
    return coords


def regularize_rboxes(rboxes):
    """Regularize rotated bounding boxes to range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (torch.Tensor): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]:
    """Convert masks to segments using contour detection.

    Args:
        masks (np.ndarray | torch.Tensor): Binary masks with shape (batch_size, 160, 160).
        strategy (str): Segmentation strategy, either 'all' or 'largest'.

    Returns:
        (list): List of segment masks as float32 arrays.
    """
    from ultralytics.data.converter import merge_multi_segment

    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """Convert a batch of FP32 torch tensors to NumPy uint8 arrays, changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch with shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch with shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).byte().cpu().numpy()


def clean_str(s):
    """Clean a string by replacing special characters with '_' character.

    Args:
        s (str): A string needing special characters replaced.

    Returns:
        (str): A string with special characters replaced by an underscore _.
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)


def empty_like(x):
    """Create empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)
