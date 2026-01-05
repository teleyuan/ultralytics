from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults


class Heatmap(ObjectCounter):
    """
    热力图(Heatmap)类：基于目标轨迹在实时视频流中绘制热力图

    该类继承自ObjectCounter类，用于生成和可视化视频流中目标移动的热力图。
    它使用追踪的目标位置，随时间累积生成热力图效果，展示目标活动密集区域。

    属性:
        initialized (bool): 标记热力图是否已初始化
        colormap (int): 用于热力图可视化的OpenCV颜色映射
        heatmap (np.ndarray): 存储累积热力图数据的数组
        annotator (SolutionAnnotator): 用于在图像上绘制标注的对象

    方法:
        heatmap_effect: 为给定边界框计算并更新热力图效果
        process: 为每一帧生成并应用热力图效果

    使用示例:
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolo11n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化Heatmap类，用于基于目标轨迹生成实时视频流热力图

        Args:
            **kwargs (Any): 传递给父类ObjectCounter的关键字参数
        """
        super().__init__(**kwargs)

        self.initialized = False  # 热力图初始化标志
        if self.region is not None:  # 检查用户是否提供了区域坐标
            self.initialize_region()

        # 存储颜色映射
        self.colormap = self.CFG["colormap"]
        self.heatmap = None

    def heatmap_effect(self, box: list[float]) -> None:
        """
        高效计算热力图区域和效果位置以应用颜色映射

        该方法在边界框区域内累积热力图强度，使用圆形区域来平滑热力图效果。
        通过向量化计算距离，高效地更新热力图数值。

        Args:
            box (list[float]): 边界框坐标 [x0, y0, x1, y1]
        """
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # 创建感兴趣区域(ROI)的网格，用于向量化距离计算
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # 计算到中心的平方距离
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # 创建半径内的点的掩码
        within_radius = dist_squared <= radius_squared

        # 在单个向量化操作中仅更新边界框内的值
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        使用Ultralytics追踪为每一帧生成热力图

        该方法实现完整的热力图生成流程：
        1. 初始化热力图数组（首次调用时）
        2. 提取目标追踪信息
        3. 为每个检测到的目标应用热力图效果
        4. 如果定义了区域，执行目标计数
        5. 归一化热力图并应用颜色映射
        6. 将热力图叠加到原始图像上

        Args:
            im0 (np.ndarray): 待处理的输入图像数组

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - in_count: 进入区域的目标计数
                - out_count: 离开区域的目标计数
                - classwise_count: 每个类别的目标计数字典
                - total_tracks: 追踪的目标总数
        """
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
            self.initialized = True  # 仅初始化热力图一次

        self.extract_tracks(im0)  # 提取轨迹
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        # 遍历边界框、追踪ID和类别索引
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # 为边界框应用热力图效果
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)  # 存储追踪历史
                # 获取前一位置（如果可用）
                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # 目标计数

        plot_im = self.annotator.result()
        if self.region is not None:
            self.display_counts(plot_im)  # 在帧上显示计数

        # 归一化热力图，应用颜色映射并与原始图像组合
        if self.track_data.is_track:
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)
            plot_im = cv2.addWeighted(plot_im, 0.5, colored_heatmap, 0.5, 0)

        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
