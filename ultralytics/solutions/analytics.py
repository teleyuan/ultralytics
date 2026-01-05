from __future__ import annotations

from itertools import cycle
from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionResults  # Import a parent class
from ultralytics.utils import plt_settings


class Analytics(BaseSolution):
    """
    数据分析(Analytics)类：创建和更新各种类型的图表用于视觉分析

    该类扩展BaseSolution以提供基于目标检测和追踪数据生成折线图、柱状图、饼图和面积图的功能。

    核心功能：
    1. 支持多种图表类型（折线图、柱状图、饼图、面积图）
    2. 实时更新图表数据
    3. 类别级统计分析
    4. 自定义图表样式和颜色

    属性:
        type (str): 要生成的分析图表类型（'line'、'bar'、'pie'或'area'）
        x_label (str): x轴标签
        y_label (str): y轴标签
        bg_color (str): 图表帧的背景颜色
        fg_color (str): 图表帧的前景颜色
        title (str): 图表窗口标题
        max_points (int): 图表上显示的最大数据点数量
        fontsize (int): 文本显示的字体大小
        color_cycle (cycle): 图表颜色的循环迭代器
        total_counts (int): 检测到的目标总计数（用于折线图）
        clswise_count (dict[str, int]): 类别级目标计数字典
        fig (Figure): Matplotlib图形对象
        ax (Axes): Matplotlib坐标轴对象
        canvas (FigureCanvasAgg): 渲染图表的画布
        lines (dict): 存储面积图线条对象的字典
        color_mapping (dict[str, str]): 将类别标签映射到颜色的字典，用于一致的可视化

    方法:
        process: 处理图像数据并更新图表
        update_graph: 使用新数据点更新图表

    使用示例:
        >>> analytics = Analytics(analytics_type="line")
        >>> frame = cv2.imread("image.jpg")
        >>> results = analytics.process(frame, frame_number=1)
        >>> cv2.imshow("Analytics", results.plot_im)
    """

    @plt_settings()
    def __init__(self, **kwargs: Any) -> None:
        """
        使用各种图表类型初始化Analytics类用于视觉数据表示

        初始化流程：
        1. 调用父类初始化
        2. 导入matplotlib相关模块
        3. 设置图表类型和坐标轴标签
        4. 配置图表样式和颜色
        5. 初始化数据容器和缓存
        6. 根据图表类型创建对应的图形对象
        """
        super().__init__(**kwargs)

        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.type = self.CFG["analytics_type"]  # 图表类型："line"、"pie"、"bar"或"area"
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        # 预定义数据
        self.bg_color = "#F3F3F3"  # 帧的背景颜色
        self.fg_color = "#111E68"  # 帧的前景颜色
        self.title = "Ultralytics Solutions"  # 窗口名称
        self.max_points = 45  # 窗口上绘制的最大点数
        self.fontsize = 25  # 显示的文本字体大小
        figsize = self.CFG["figsize"]  # 输出尺寸，例如 (12.8, 7.2) -> 1280x720
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # 存储折线图的总计数
        self.clswise_count = {}  # 类别级计数字典
        self.update_every = kwargs.get("update_every", 30)  # 默认每30帧更新一次图表
        self.last_plot_im = None  # 上次渲染图表的缓存

        # 确保折线图和面积图
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvasAgg(self.fig)  # 设置公共轴属性
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # 初始化柱状图或饼图
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvasAgg(self.fig)  # 设置公共轴属性
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":  # 确保饼图是圆形的
                self.ax.axis("equal")

    def process(self, im0: np.ndarray, frame_number: int) -> SolutionResults:
        """
        处理图像数据并运行目标追踪以更新分析图表

        处理流程：
        1. 提取追踪目标
        2. 根据图表类型进行计数：
           - 折线图：累加总计数
           - 其他类型：按类别统计
        3. 判断是否需要更新图表（每N帧更新一次）
        4. 更新图表并缓存结果
        5. 返回处理结果

        Args:
            im0 (np.ndarray): 待处理的输入图像
            frame_number (int): 用于绘制数据的视频帧号

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_tracks: 追踪的目标总数（int）
                - classwise_count: 按类别的目标计数（dict）

        Raises:
            ValueError: 如果指定了不支持的图表类型

        使用示例:
            >>> analytics = Analytics(analytics_type="line")
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> results = analytics.process(frame, frame_number=1)
        """
        self.extract_tracks(im0)  # 提取追踪目标
        if self.type == "line":
            for _ in self.boxes:
                self.total_counts += 1
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(frame_number=frame_number)
            plot_im = self.last_plot_im
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            from collections import Counter

            self.clswise_count = Counter(self.names[int(cls)] for cls in self.clss)
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(
                    frame_number=frame_number, count_dict=self.clswise_count, plot=self.type
                )
            plot_im = self.last_plot_im
        else:
            raise ValueError(f"不支持的analytics_type='{self.type}'。支持的类型：line, bar, pie, area。")

        # 返回结果供下游使用
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), classwise_count=self.clswise_count)

    def update_graph(
        self, frame_number: int, count_dict: dict[str, int] | None = None, plot: str = "line"
    ) -> np.ndarray:
        """
        使用新数据更新单个或多个类别的图表

        该方法根据不同的图表类型实现不同的更新逻辑：
        - 折线图：单条线的数据点更新
        - 面积图：多条线的填充区域更新
        - 柱状图：类别计数的柱状显示
        - 饼图：类别占比的扇形显示

        处理流程：
        1. 根据图表类型更新数据：
           a. 折线图：追加新数据点并限制最大点数
           b. 面积图：多条线的数据更新和填充
           c. 柱状图：清除旧数据并绘制新柱状图
           d. 饼图：计算百分比并绘制扇形
        2. 设置图表样式（背景、网格、标题、标签）
        3. 添加和格式化图例
        4. 重绘图表并转换为OpenCV图像格式
        5. 显示并返回更新后的图像

        Args:
            frame_number (int): 当前帧号
            count_dict (dict[str, int], optional): 以类别名称为键、计数为值的字典，用于多类别。
                如果为None，则更新单条折线图
            plot (str): 图表类型。选项有'line'、'bar'、'pie'或'area'

        Returns:
            (np.ndarray): 包含图表的更新后图像

        使用示例:
            >>> analytics = Analytics(analytics_type="bar")
            >>> frame_num = 10
            >>> results_dict = {"person": 5, "car": 3}
            >>> updated_image = analytics.update_graph(frame_num, results_dict, plot="bar")
        """
        if count_dict is None:
            # 单条线更新
            x_data = np.append(self.line.get_xdata(), float(frame_number))
            y_data = np.append(self.line.get_ydata(), float(self.total_counts))

            if len(x_data) > self.max_points:
                x_data, y_data = x_data[-self.max_points :], y_data[-self.max_points :]

            self.line.set_data(x_data, y_data)
            self.line.set_label("Counts")
            self.line.set_color("#7b0068")  # 粉色
            self.line.set_marker("*")
            self.line.set_markersize(self.line_width * 5)
        else:
            labels = list(count_dict.keys())
            counts = list(count_dict.values())
            if plot == "area":
                color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])
                # 多条线或面积更新
                x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
                y_data_dict = {key: np.array([]) for key in count_dict.keys()}
                if self.ax.lines:
                    for line, key in zip(self.ax.lines, count_dict.keys()):
                        y_data_dict[key] = line.get_ydata()

                x_data = np.append(x_data, float(frame_number))
                max_length = len(x_data)
                for key in count_dict.keys():
                    y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                    if len(y_data_dict[key]) < max_length:
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])))
                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict.keys():
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()
                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.55)
                    self.ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        marker="o",
                        markersize=self.line_width * 5,
                        label=f"{key} Data Points",
                    )
            elif plot == "bar":
                self.ax.clear()  # 清除柱状图数据
                for label in labels:  # 将标签映射到颜色
                    if label not in self.color_mapping:
                        self.color_mapping[label] = next(self.color_cycle)
                colors = [self.color_mapping[label] for label in labels]
                bars = self.ax.bar(labels, counts, color=colors)
                for bar, count in zip(bars, counts):
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha="center",
                        va="bottom",
                        color=self.fg_color,
                    )
                # 使用柱状图的标签创建图例
                for bar, label in zip(bars, labels):
                    bar.set_label(label)  # 为每个柱分配标签
                self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)
            elif plot == "pie":
                total = sum(counts)
                percentages = [size / total * 100 for size in counts]
                self.ax.clear()

                start_angle = 90
                # 创建饼图并使用百分比创建图例标签
                wedges, _ = self.ax.pie(
                    counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
                )
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

                # 使用扇形和手动创建的标签分配图例
                self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                self.fig.subplots_adjust(left=0.1, right=0.75)  # 调整布局以适应图例

        # 公共图表设置
        self.ax.set_facecolor("#f0f0f0")  # 设置为浅灰色或其他您喜欢的颜色
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)  # 显示网格以获得更多数据洞察
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

        # 添加和格式化图例
        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.bg_color)
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        # 重绘图表、更新视图、捕获并显示更新后的图
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display_output(im0)

        return im0  # 返回图像
