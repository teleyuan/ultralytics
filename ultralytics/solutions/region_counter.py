from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """
    区域计数器(RegionCounter)类：在视频流中对用户定义区域内的目标进行实时计数

    该类继承自BaseSolution类，提供在视频帧中定义多个多边形区域、追踪目标并计数通过每个定义区域的目标的功能。
    适用于需要在指定区域进行计数的应用，例如监控区域、分段部分或多个感兴趣区域。

    核心功能：
    1. 支持定义多个计数区域（多边形）
    2. 每个区域可独立命名和配色
    3. 实时追踪目标并判断是否在区域内
    4. 为每个区域单独统计目标数量
    5. 可视化显示各区域的计数结果

    属性:
        region_template (dict): 创建新计数区域的模板，包含默认属性如名称、多边形坐标和显示颜色
        counting_regions (list): 存储所有定义区域的列表，每个条目基于region_template并包含特定的区域设置
        region_counts (dict): 存储每个命名区域的目标计数的字典

    方法:
        add_region: 添加具有指定属性的新计数区域
        process: 处理视频帧以计数每个区域中的目标
        initialize_regions: 初始化区域以计数每个区域中的目标

    使用示例:
        >>> from ultralytics.solutions import RegionCounter
        >>> counter = RegionCounter()
        >>> counter.add_region("区域1", [(100, 100), (200, 100), (200, 200), (100, 200)], (255, 0, 0), (255, 255, 255))
        >>> results = counter.process(frame)
        >>> print(f"追踪总数: {results.total_tracks}")
        >>> print(f"区域计数: {results.region_counts}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化RegionCounter，用于在用户定义的区域中实时计数目标

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - region: 区域字典，键为区域名称，值为多边形坐标列表
                - model: YOLO模型路径
                - line_width: 绘制线条的宽度
        """
        super().__init__(**kwargs)
        self.region_template = {
            "name": "默认区域",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        """
        基于提供的模板和特定属性向计数列表添加新区域

        该方法创建一个新的计数区域，并将其添加到区域列表中。每个区域都有独立的
        名称、多边形边界、颜色设置和计数器。

        Args:
            name (str): 分配给新区域的名称
            polygon_points (list[tuple]): 定义区域多边形的(x, y)坐标列表
            region_color (tuple[int, int, int]): 区域可视化的BGR颜色
            text_color (tuple[int, int, int]): 区域内文本的BGR颜色

        Returns:
            (dict[str, Any]): 区域信息，包括名称、多边形和显示颜色

        使用示例:
            >>> counter = RegionCounter()
            >>> region = counter.add_region("入口", [(0, 0), (100, 0), (100, 100), (0, 100)], (0, 255, 0), (255, 255, 255))
        """
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """
        从self.region初始化区域（仅执行一次）

        该方法将配置中的区域定义转换为计数区域对象。如果region未定义为字典，
        会自动将其转换为字典格式。为每个区域分配颜色并创建预处理的多边形对象以提高性能。
        """
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # 确保self.region被初始化并结构化为字典
            self.region = {"区域#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        处理输入帧以检测和计数每个定义区域内的目标

        该方法实现完整的多区域计数流程：
        1. 提取当前帧的目标追踪信息
        2. 初始化标注器
        3. 遍历所有检测到的目标：
           - 绘制边界框和标签
           - 计算目标边界框的质心坐标
           - 检查质心是否在任何定义的区域内
           - 如果在区域内，增加该区域的计数
        4. 为每个区域绘制多边形边界和计数标签
        5. 重置每个区域的计数以准备处理下一帧

        Args:
            im0 (np.ndarray): 输入图像帧，将在其上标注目标和区域

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_tracks: 追踪的目标总数
                - region_counts: 每个区域的目标计数字典

        使用示例:
            >>> counter = RegionCounter()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = counter.process(frame)
            >>> print(f"区域计数: {results.region_counts}")
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # 显示区域计数
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
            region["counts"] = 0  # 为下一帧重置计数
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
