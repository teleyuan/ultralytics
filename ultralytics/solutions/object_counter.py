from __future__ import annotations

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """
    目标计数器(ObjectCounter)类：基于追踪在实时视频流中管理目标计数

    该类继承自BaseSolution类，提供在视频流中计数进出指定区域的目标的功能。
    支持多边形区域和线性区域两种计数方式，可用于人流统计、车辆计数等场景。

    属性:
        in_count (int): 进入区域的目标计数器
        out_count (int): 离开区域的目标计数器
        counted_ids (list[int]): 已计数目标的ID列表
        classwise_count (dict[str, dict[str, int]]): 按目标类别分类的计数字典
        region_initialized (bool): 标记计数区域是否已初始化
        show_in (bool): 控制是否显示进入计数
        show_out (bool): 控制是否显示离开计数
        margin (int): 显示计数时背景矩形的边距

    方法:
        count_objects: 基于轨迹在多边形或线性区域内计数目标
        display_counts: 在帧上显示目标计数
        process: 处理输入数据并更新计数

    使用示例:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = counter.process(frame)
        >>> print(f"进入计数: {counter.in_count}, 离开计数: {counter.out_count}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化ObjectCounter类，用于视频流中的实时目标计数
        """
        super().__init__(**kwargs)

        self.in_count = 0  # 进入区域的目标计数器
        self.out_count = 0  # 离开区域的目标计数器
        self.counted_ids = []  # 已计数目标的ID列表
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # 按类别分类的计数字典
        self.region_initialized = False  # 标记区域是否已初始化

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2  # 缩放背景矩形大小以正确显示计数

    def count_objects(
        self,
        current_centroid: tuple[float, float],
        track_id: int,
        prev_position: tuple[float, float] | None,
        cls: int,
    ) -> None:
        """
        基于轨迹在多边形或线性区域内计数目标

        该方法通过比较目标当前位置和前一帧位置，判断目标的运动方向（进入或离开），
        从而实现目标计数。支持线性区域（直线）和多边形区域两种计数方式。

        Args:
            current_centroid (tuple[float, float]): 当前帧中的质心坐标(x, y)
            track_id (int): 追踪目标的唯一标识符
            prev_position (tuple[float, float], optional): 追踪目标在上一帧的位置坐标(x, y)
            cls (int): 用于更新分类计数的类别索引

        使用示例:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id_num = 1
            >>> previous_position = (120, 220)
            >>> class_to_count = 0  # 在COCO模型中，类别0 = 人
            >>> counter.count_objects((140, 240), track_id_num, previous_position, class_to_count)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:  # 线性区域（定义为线段）
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                # 判断区域的方向（垂直或水平）
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # 垂直区域：比较x坐标判断方向
                    if current_centroid[0] > prev_position[0]:  # 向右移动
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:  # 向左移动
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                # 水平区域：比较y坐标判断方向
                elif current_centroid[1] > prev_position[1]:  # 向下移动
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # 向上移动
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # 多边形区域
            if self.r_s.contains(self.Point(current_centroid)):
                # 判断垂直或水平多边形的运动方向
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (region_width < region_height and current_centroid[0] > prev_position[0]) or (
                    region_width >= region_height and current_centroid[1] > prev_position[1]
                ):  # 向右或向下移动
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # 向左或向上移动
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def display_counts(self, plot_im) -> None:
        """
        在输入图像或帧上显示目标计数

        Args:
            plot_im (np.ndarray): 要显示计数的图像或帧

        使用示例:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if value["IN"] != 0 or (value["OUT"] != 0 and (self.show_in or self.show_out))
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0) -> SolutionResults:
        """
        处理输入数据（帧或目标轨迹）并更新目标计数

        该方法实现完整的目标计数流程：
        1. 初始化计数区域
        2. 提取追踪轨迹
        3. 绘制边界框和区域
        4. 更新目标计数
        5. 在输入图像上显示结果

        Args:
            im0 (np.ndarray): 待处理的输入图像或帧

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - in_count: 进入区域的目标计数
                - out_count: 离开区域的目标计数
                - classwise_count: 每个类别的目标计数字典
                - total_tracks: 追踪的目标总数

        使用示例:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = counter.process(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  # 提取轨迹
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # 绘制区域

        # 遍历边界框、追踪ID和类别索引
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 绘制边界框和计数区域
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # 存储追踪历史

            # 存储追踪的前一位置用于目标计数
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # 目标计数

        plot_im = self.annotator.result()
        self.display_counts(plot_im)  # 在帧上显示计数
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
