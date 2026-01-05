"""
解决方案配置模块

本模块提供了 Ultralytics 所有解决方案的统一配置管理类。通过集中管理配置参数，
确保各个解决方案模块的配置一致性和可维护性。

主要功能:
    - 提供类型安全的配置参数定义
    - 支持动态更新配置参数
    - 为所有解决方案提供默认配置值
    - 参数验证和错误提示

典型应用场景:
    - 初始化解决方案实例时提供配置
    - 运行时动态调整解决方案参数
    - 跨解决方案共享通用配置
"""

from __future__ import annotations  # 启用延迟类型注解评估，支持 Python 3.9+ 的新式类型提示

from dataclasses import dataclass, field  # 用于创建数据类，简化配置类定义
from typing import Any  # 泛型类型提示

import cv2  # OpenCV 库，用于获取颜色映射常量


@dataclass
class SolutionConfig:
    """
    管理Ultralytics视觉AI解决方案的配置参数

    SolutionConfig类作为所有Ultralytics解决方案模块的集中式配置容器：
    https://docs.ultralytics.com/solutions/#solutions。它利用Python的`dataclass`提供清晰、类型安全和可维护的参数定义。

    属性:
        source (str, optional): 输入源路径（视频、RTSP等）。仅可用于Solutions CLI
        model (str, optional): 用于推理的Ultralytics YOLO模型路径
        classes (list[int], optional): 用于过滤检测的类别索引列表
        show_conf (bool): 是否在视觉输出上显示置信度分数
        show_labels (bool): 是否在视觉输出上显示类别标签
        region (list[tuple[int, int]], optional): 用于物体计数的多边形区域或线段
        colormap (int, optional): 用于视觉叠加的OpenCV颜色映射常量（例如cv2.COLORMAP_JET）
        show_in (bool): 是否显示进入区域的物体计数
        show_out (bool): 是否显示离开区域的物体计数
        up_angle (float): 基于姿态的运动监测中使用的上角度阈值
        down_angle (int): 基于姿态的运动监测中使用的下角度阈值
        kpts (list[int]): 要监测的关键点索引，例如用于姿态分析
        analytics_type (str): 要执行的分析类型（"line"、"area"、"bar"、"pie"等）
        figsize (tuple[int, int], optional): 用于分析图的matplotlib图形尺寸（宽度、高度）
        blur_ratio (float): 用于模糊视频帧中物体的比率（0.0到1.0）
        vision_point (tuple[int, int]): 用于方向追踪或透视绘制的参考点
        crop_dir (str): 保存裁剪检测图像的目录路径
        json_file (str): 包含停车区域数据的JSON文件路径
        line_width (int): 视觉显示的宽度，例如边界框、关键点和计数
        records (int): 发送电子邮件告警的检测记录数
        fps (float): 用于速度估计计算的帧率（帧/秒）
        max_hist (int): 每个追踪物体存储的最大历史点数或状态数，用于速度估计
        meter_per_pixel (float): 真实世界测量的比例，用于速度或距离计算
        max_speed (int): 用于视觉告警或约束的最大速度限制（例如km/h或mph）
        show (bool): 是否在屏幕上显示视觉输出
        iou (float): 用于检测过滤的交并比阈值
        conf (float): 保留预测的置信度阈值
        device (str, optional): 运行推理的设备（例如'cpu'、'0'表示CUDA GPU）
        max_det (int): 每个视频帧允许的最大检测数
        half (bool): 是否使用FP16精度（需要支持的CUDA设备）
        tracker (str): 追踪配置YAML文件路径（例如'botsort.yaml'）
        verbose (bool): 启用详细日志输出用于调试或诊断
        data (str): 用于相似度搜索的图像目录路径

    方法:
        update: 使用用户定义的关键字参数更新配置，并在无效键上引发错误

    使用示例:
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo11n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """

    # ==================== 基础配置参数 ====================
    source: str | None = None  # 输入源路径（视频、RTSP流等），仅用于 CLI 模式
    model: str | None = None  # YOLO 模型路径或名称
    classes: list[int] | None = None  # 要检测的类别索引列表，None 表示检测所有类别
    show_conf: bool = True  # 是否在可视化输出中显示置信度分数
    show_labels: bool = True  # 是否在可视化输出中显示类别标签
    region: list[tuple[int, int]] | None = None  # 多边形区域或线段，用于物体计数等功能

    # ==================== 视觉效果配置 ====================
    colormap: int | None = cv2.COLORMAP_DEEPGREEN  # OpenCV 颜色映射，用于热力图等可视化
    show_in: bool = True  # 是否显示进入区域的物体计数
    show_out: bool = True  # 是否显示离开区域的物体计数

    # ==================== 姿态检测配置（AI Gym） ====================
    up_angle: float = 145.0  # 运动"向上"状态的角度阈值（度）
    down_angle: int = 90  # 运动"向下"状态的角度阈值（度）
    kpts: list[int] = field(default_factory=lambda: [6, 8, 10])  # 要监测的关键点索引列表（默认：肩、肘、腕）

    # ==================== 数据分析配置（Analytics） ====================
    analytics_type: str = "line"  # 分析图表类型："line"(折线图), "area"(面积图), "bar"(柱状图), "pie"(饼图)
    figsize: tuple[int, int] | None = (12.8, 7.2)  # Matplotlib 图表尺寸（英寸），对应 1280x720 像素

    # ==================== 图像处理配置 ====================
    blur_ratio: float = 0.5  # 模糊比率（0.0-1.0），用于 ObjectBlurrer
    vision_point: tuple[int, int] = (20, 20)  # 视觉焦点参考点坐标（x, y），用于 VisionEye
    crop_dir: str = "cropped-detections"  # 裁剪目标的保存目录，用于 ObjectCropper

    # ==================== 停车管理配置 ====================
    json_file: str = None  # 停车位区域定义 JSON 文件路径

    # ==================== 显示配置 ====================
    line_width: int = 2  # 边界框、关键点、计数文本等的线宽

    # ==================== 安全告警配置 ====================
    records: int = 5  # 触发邮件告警的检测记录数阈值

    # ==================== 速度估计配置 ====================
    fps: float = 30.0  # 视频帧率（帧/秒），用于速度估计的时间计算
    max_hist: int = 5  # 每个追踪对象保存的最大历史帧数，用于速度估计
    meter_per_pixel: float = 0.05  # 每像素对应的真实世界距离（米），取决于相机参数
    max_speed: int = 120  # 最大速度限制（km/h），超过此值的速度将被限制

    # ==================== 显示控制 ====================
    show: bool = False  # 是否在屏幕上显示可视化输出

    # ==================== 模型推理配置 ====================
    iou: float = 0.7  # NMS（非极大值抑制）的 IoU 阈值
    conf: float = 0.25  # 检测置信度阈值，低于此值的检测将被过滤
    device: str | None = None  # 推理设备（'cpu', '0', 'cuda:0' 等）
    max_det: int = 300  # 每帧最大检测数量
    half: bool = False  # 是否使用 FP16 半精度推理（需要 CUDA 支持）

    # ==================== 追踪配置 ====================
    tracker: str = "botsort.yaml"  # 追踪器配置文件路径

    # ==================== 调试配置 ====================
    verbose: bool = True  # 是否启用详细日志输出

    # ==================== 相似度搜索配置 ====================
    data: str = "images"  # 图像目录路径，用于 VisualAISearch

    def update(self, **kwargs: Any):
        """
        使用关键字参数提供的新值更新配置参数

        该方法允许动态更新配置对象的属性值，并在提供无效参数时抛出错误。

        Args:
            **kwargs (Any): 要更新的配置参数键值对

        Returns:
            (SolutionConfig): 返回self以支持链式调用

        Raises:
            ValueError: 当提供的键不是有效的配置参数时
        """
        # 遍历所有传入的键值对参数
        for key, value in kwargs.items():
            # 检查该配置参数是否存在于数据类中
            if hasattr(self, key):
                setattr(self, key, value)  # 更新参数值
            else:
                # 如果参数不存在，抛出错误并提供文档链接
                url = "https://docs.ultralytics.com/solutions/#solutions-arguments"
                raise ValueError(f"{key} is not a valid solution argument, see {url}")

        return self  # 返回 self 以支持链式调用
