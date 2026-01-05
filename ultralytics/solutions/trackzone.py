from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class TrackZone(BaseSolution):
    """
    追踪区域(TrackZone)类：在视频流中管理基于区域的目标追踪

    该类继承自BaseSolution类，提供在由多边形区域定义的特定区域内追踪目标的功能。
    区域外的目标将被排除在追踪之外。主要应用于需要关注特定区域的场景，如禁区监控、特定区域活动分析等。

    核心功能：
    1. 定义多边形追踪区域
    2. 创建区域掩码
    3. 仅追踪区域内的目标
    4. 可视化显示区域边界和追踪结果

    属性:
        region (np.ndarray): 用于追踪的多边形区域，表示为点的凸包
        mask (np.ndarray): 区域掩码，用于过滤区域外的目标
        line_width (int): 用于绘制边界框和区域边界的线条宽度
        names (list[str]): 模型可以检测的类别名称列表
        boxes (list[np.ndarray]): 追踪目标的边界框
        track_ids (list[int]): 每个追踪目标的唯一标识符
        clss (list[int]): 追踪目标的类别索引

    方法:
        process: 处理视频的每一帧，应用基于区域的追踪
        extract_tracks: 从输入帧提取追踪信息
        display_output: 显示处理后的输出

    使用示例:
        >>> from ultralytics.solutions import TrackZone
        >>> tracker = TrackZone(region=[(100, 100), (500, 100), (500, 400), (100, 400)])
        >>> frame = cv2.imread("frame.jpg")
        >>> results = tracker.process(frame)
        >>> cv2.imshow("追踪帧", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化TrackZone类，用于在视频流中的定义区域内追踪目标

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - region: 多边形区域的顶点坐标列表
                - model: YOLO模型路径
                - line_width: 绘制线条的宽度
        """
        super().__init__(**kwargs)
        default_region = [(75, 75), (565, 75), (565, 285), (75, 285)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))
        self.mask = None

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        处理输入帧以在定义的区域内追踪目标

        该方法实现完整的区域追踪流程：
        1. 初始化标注器
        2. 创建区域掩码（首次调用时）：
           - 创建与图像大小相同的零掩码
           - 使用fillPoly填充多边形区域为255
        3. 使用掩码过滤图像，仅保留区域内的内容
        4. 从掩码图像中提取追踪轨迹（区域外目标被忽略）
        5. 在原图上绘制区域边界
        6. 为区域内的每个目标绘制边界框和标签
        7. 返回处理结果

        Args:
            im0 (np.ndarray): 待处理的输入图像或帧

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_tracks: 定义区域内追踪的目标总数

        使用示例:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = tracker.process(frame)
            >>> print(f"区域内追踪到 {results.total_tracks} 个目标")
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        if self.mask is None:  # 创建区域掩码
            self.mask = np.zeros_like(im0[:, :, 0])
            cv2.fillPoly(self.mask, [self.region], 255)
        masked_frame = cv2.bitwise_and(im0, im0, mask=self.mask)
        self.extract_tracks(masked_frame)

        # 绘制区域边界
        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # 遍历边界框、追踪ID、类别索引列表并绘制边界框
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, track_id=track_id), color=colors(track_id, True)
            )

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
