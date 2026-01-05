from collections import deque
from math import sqrt
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class SpeedEstimator(BaseSolution):
    """
    速度估计器(SpeedEstimator)类：基于目标轨迹在实时视频流中估计目标速度

    该类继承自BaseSolution类，使用视频流中的追踪数据提供目标速度估计功能。
    速度计算基于像素位移随时间的变化，并通过可配置的米/像素比例因子转换为真实世界单位。

    核心原理：
    1. 追踪目标在多帧中的位置历史
    2. 计算起始点和结束点之间的像素距离
    3. 根据帧率计算时间间隔
    4. 将像素距离转换为实际距离（米）
    5. 使用公式 速度 = 距离/时间 * 3.6 转换为 km/h

    属性:
        fps (float): 视频帧率，用于时间计算
        frame_count (int): 全局帧计数器，用于追踪时间信息
        trk_frame_ids (dict): 将追踪ID映射到其首次出现的帧索引
        spd (dict): 每个目标的最终速度(km/h)，一旦锁定不再更新
        trk_hist (dict): 将追踪ID映射到位置历史的双端队列
        locked_ids (set): 速度已确定的追踪ID集合
        max_hist (int): 计算速度前所需的最小帧历史数量
        meter_per_pixel (float): 一个像素代表的实际世界距离（米），用于场景比例转换
        max_speed (int): 允许的最大目标速度，超过此值将被限制

    方法:
        process: 处理输入帧，基于追踪数据估计目标速度
        store_tracking_history: 存储目标的追踪历史
        extract_tracks: 从当前帧提取追踪轨迹
        display_output: 显示带有标注的输出

    使用示例:
        >>> from ultralytics.solutions import SpeedEstimator
        >>> estimator = SpeedEstimator(meter_per_pixel=0.04, max_speed=120)
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("速度估计", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化SpeedEstimator对象，配置速度估计参数和数据结构

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - fps: 视频帧率（默认25）
                - max_hist: 速度计算所需的历史帧数（默认30）
                - meter_per_pixel: 像素到米的转换比例（需根据相机参数设定）
                - max_speed: 最大速度限制（km/h）
        """
        super().__init__(**kwargs)

        self.fps = self.CFG["fps"]  # 视频帧率，用于时间计算
        self.frame_count = 0  # 全局帧计数器
        self.trk_frame_ids = {}  # 追踪ID → 首次出现的帧索引
        self.spd = {}  # 每个目标的最终速度(km/h)，一旦锁定不再更新
        self.trk_hist = {}  # 追踪ID → 位置历史的双端队列 (时间, 位置)
        self.locked_ids = set()  # 速度已确定的追踪ID集合
        self.max_hist = self.CFG["max_hist"]  # 计算速度前所需的帧历史数量
        self.meter_per_pixel = self.CFG["meter_per_pixel"]  # 场景比例，取决于相机参数
        self.max_speed = self.CFG["max_speed"]  # 最大速度限制

    def process(self, im0) -> SolutionResults:
        """
        处理输入帧，基于追踪数据估计目标速度

        该方法实现完整的速度估计流程：
        1. 提取当前帧的目标追踪信息
        2. 更新每个目标的位置历史
        3. 当积累足够历史帧后，计算速度：
           - 获取起始和结束位置
           - 计算像素距离：sqrt((x1-x0)² + (y1-y0)²)
           - 转换为实际距离：像素距离 × 米/像素
           - 计算速度：(距离/时间) × 3.6 转换为 km/h
        4. 锁定已计算的速度，释放历史数据
        5. 在图像上标注速度信息

        Args:
            im0 (np.ndarray): 待处理的输入图像，形状为 (H, W, C)，OpenCV BGR格式

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 带有速度标注的处理后图像
                - total_tracks: 追踪的目标数量

        使用示例:
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, _, _ in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:  # 如果是新追踪目标，初始化历史记录
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
                self.trk_frame_ids[track_id] = self.frame_count

            if track_id not in self.locked_ids:  # 在速度锁定前持续更新历史
                trk_hist = self.trk_hist[track_id]
                trk_hist.append(self.track_line[-1])

                # 一旦收集到足够的历史数据，计算并锁定速度
                if len(trk_hist) == self.max_hist:
                    p0, p1 = trk_hist[0], trk_hist[-1]  # 轨迹的起始点和结束点
                    dt = (self.frame_count - self.trk_frame_ids[track_id]) / self.fps  # 时间间隔（秒）
                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]  # 像素位移
                        pixel_distance = sqrt(dx * dx + dy * dy)  # 计算像素距离
                        meters = pixel_distance * self.meter_per_pixel  # 转换为米
                        self.spd[track_id] = int(
                            min((meters / dt) * 3.6, self.max_speed)
                        )  # 转换为 km/h 并存储最终速度（限制在最大速度内）
                        self.locked_ids.add(track_id)  # 防止进一步更新
                        self.trk_hist.pop(track_id, None)  # 释放内存
                        self.trk_frame_ids.pop(track_id, None)  # 移除帧起始引用

            if track_id in self.spd:
                speed_label = f"{self.spd[track_id]} km/h"
                annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边界框和速度标签

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回包含处理图像和追踪摘要的结果
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
