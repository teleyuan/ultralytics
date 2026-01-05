from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class QueueManager(BaseSolution):
    """
    队列管理器(QueueManager)类：基于目标轨迹在实时视频流中管理队列计数

    该类继承自BaseSolution类，提供在视频帧中追踪和计数指定区域内目标的功能。
    主要应用于排队场景，如银行、超市、机场等场所的队列长度统计。

    核心功能：
    1. 定义队列区域（矩形或多边形）
    2. 追踪进入队列区域的目标
    3. 实时统计队列中的目标数量
    4. 可视化显示队列区域和计数结果

    属性:
        counts (int): 队列中目标的当前计数
        rect_color (tuple[int, int, int]): 用于绘制队列区域矩形的BGR颜色元组
        region_length (int): 定义队列区域的点数
        track_line (list[tuple[int, int]]): 轨迹线坐标列表
        track_history (dict[int, list[tuple[int, int]]]): 存储每个目标追踪历史的字典

    方法:
        initialize_region: 初始化队列区域
        process: 处理队列管理的单帧视频
        extract_tracks: 从当前帧中提取目标轨迹
        store_tracking_history: 存储目标的追踪历史
        display_output: 显示处理后的输出

    使用示例:
        >>> import cv2
        >>> from ultralytics.solutions import QueueManager
        >>> cap = cv2.VideoCapture("path/to/video.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        ...     success, im0 = cap.read()
        ...     if not success:
        ...         break
        ...     results = queue_manager.process(im0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化QueueManager，设置视频流中目标追踪和计数的参数

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - region: 队列区域坐标列表
                - model: YOLO模型路径
                - line_width: 绘制线条的宽度
        """
        super().__init__(**kwargs)
        self.initialize_region()  # 初始化队列区域
        self.counts = 0  # 队列计数信息
        self.rect_color = (255, 255, 255)  # 可视化的矩形颜色
        self.region_length = len(self.region)  # 存储区域长度以供进一步使用

    def process(self, im0) -> SolutionResults:
        """
        处理单帧视频的队列管理

        该方法实现完整的队列管理流程：
        1. 重置当前帧的计数
        2. 提取当前帧的目标追踪轨迹
        3. 绘制队列区域边界
        4. 遍历所有检测到的目标：
           - 绘制边界框和标签
           - 存储追踪历史
           - 检查目标是否在队列区域内
           - 如果在区域内，增加计数
        5. 在图像上显示队列计数
        6. 返回处理结果

        Args:
            im0 (np.ndarray): 用于处理的输入图像，通常是视频流中的一帧

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - queue_count: 队列中的目标数量
                - total_tracks: 追踪的目标总数

        使用示例:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = queue_manager.process(frame)
            >>> print(f"队列中有 {results.queue_count} 个目标")
        """
        self.counts = 0  # 每帧重置计数
        self.extract_tracks(im0)  # 从当前帧提取轨迹
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)  # 绘制区域

        # 遍历所有检测到的目标
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 绘制边界框和计数区域
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # 存储追踪历史

            # 缓存频繁访问的属性
            track_history = self.track_history.get(track_id, [])

            # 存储轨迹的前一个位置，并检查目标是否在计数区域内
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            # 如果区域至少有3个点、存在前一个位置且当前轨迹点在区域内，则增加计数
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # 显示队列计数
        annotator.queue_counts_display(
            f"队列计数: {self.counts}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回包含处理数据的SolutionResults对象
        return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
