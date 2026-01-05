import math
from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class DistanceCalculation(BaseSolution):
    """
    距离计算(DistanceCalculation)类：基于目标轨迹在实时视频流中计算两个目标之间的距离

    该类继承自BaseSolution类，提供在视频流中使用YOLO目标检测和追踪选择目标并计算它们之间距离的功能。
    支持通过鼠标交互选择两个目标，并实时计算和显示它们质心之间的像素距离。

    核心功能：
    1. 鼠标交互选择目标（左键选择，右键清除）
    2. 自动追踪已选择的目标
    3. 计算两个目标质心之间的欧几里得距离
    4. 可视化显示距离测量结果

    属性:
        left_mouse_count (int): 左键点击计数器
        selected_boxes (dict[int, Any]): 存储已选择边界框的字典，以追踪ID为键
        centroids (list[list[int]]): 存储已选择边界框质心的列表

    方法:
        mouse_event_for_distance: 处理鼠标事件以选择视频流中的目标
        process: 处理视频帧并计算选定目标之间的距离

    使用示例:
        >>> from ultralytics.solutions import DistanceCalculation
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = distance_calc.process(frame)
        >>> cv2.imshow("距离计算", results.plot_im)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化DistanceCalculation类，用于测量视频流中目标之间的距离

        该类通过鼠标交互方式让用户选择两个感兴趣的目标，然后自动追踪这些目标
        并实时计算它们之间的距离。
        """
        super().__init__(**kwargs)

        # 鼠标事件信息
        self.left_mouse_count = 0
        self.selected_boxes: dict[int, list[float]] = {}
        self.centroids: list[list[int]] = []  # 存储选定目标的质心

    def mouse_event_for_distance(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        处理鼠标事件以在实时视频流中选择用于距离计算的区域

        该方法允许用户通过鼠标点击选择两个目标进行距离测量：
        - 左键单击：选择目标（最多2个）
        - 右键单击：清除所有选择

        Args:
            event (int): 鼠标事件类型（例如：cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN）
            x (int): 鼠标指针的X坐标
            y (int): 鼠标指针的Y坐标
            flags (int): 与事件关联的标志（例如：cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY）
            param (Any): 传递给函数的额外参数

        使用示例:
            >>> dc = DistanceCalculation()
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def process(self, im0) -> SolutionResults:
        """
        处理视频帧并计算两个选定边界框之间的距离

        该方法实现完整的距离计算流程：
        1. 从输入帧提取追踪轨迹
        2. 标注所有边界框
        3. 更新已选择目标的边界框位置（如果它们仍在被追踪）
        4. 当选择了两个目标时：
           - 计算每个边界框的质心：((x0+x1)/2, (y0+y1)/2)
           - 使用欧几里得距离公式计算质心间距离：sqrt((x1-x0)² + (y1-y0)²)
           - 在图像上绘制距离线和数值
        5. 设置鼠标回调以支持交互式目标选择

        Args:
            im0 (np.ndarray): 待处理的输入图像帧

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_tracks: 追踪的目标总数
                - pixels_distance: 选定目标之间的像素距离

        使用示例:
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = dc.process(frame)
            >>> print(f"距离: {results.pixels_distance:.2f} 像素")
        """
        self.extract_tracks(im0)  # 提取追踪轨迹
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        pixels_distance = 0
        # 遍历边界框、追踪ID和类别索引
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, color=colors(int(cls), True), label=self.adjust_box_label(cls, conf, track_id))

            # 如果选定的目标仍在被追踪，则更新其边界框
            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # 计算选定边界框的质心
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # 计算质心之间的欧几里得距离
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []  # 为下一帧重置质心列表
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出
        if self.CFG.get("show") and self.env_check:
            cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        # 返回包含处理图像和计算指标的SolutionResults对象
        return SolutionResults(plot_im=plot_im, pixels_distance=pixels_distance, total_tracks=len(self.track_ids))
