# 导入未来版本特性支持，用于类型注解
from __future__ import annotations

# 导入数学计算模块
import math
# 导入集合工具，用于计数和默认字典
from collections import Counter, defaultdict
# 导入函数缓存装饰器，用于优化重复计算
from functools import lru_cache
# 导入类型提示工具
from typing import Any

# 导入 OpenCV 库，用于图像处理
import cv2
# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 YOLO 模型类
from ultralytics import YOLO
# 导入解决方案配置类
from ultralytics.solutions.config import SolutionConfig
# 导入工具函数和常量
from ultralytics.utils import ASSETS_URL, LOGGER, ops
# 导入检查工具函数
from ultralytics.utils.checks import check_imshow, check_requirements
# 导入标注绘图类
from ultralytics.utils.plotting import Annotator


class BaseSolution:
    """
    用于管理 Ultralytics 解决方案的基类。
    A base class for managing Ultralytics Solutions.

    该类为各种 Ultralytics 解决方案提供核心功能，包括模型加载、目标跟踪和区域初始化。
    它作为实现特定计算机视觉解决方案（如目标计数、姿态估计和分析）的基础。
    This class provides core functionality for various Ultralytics Solutions, including model loading, object tracking,
    and region initialization. It serves as the foundation for implementing specific computer vision solutions such as
    object counting, pose estimation, and analytics.

    Attributes:
        LineString: 从 shapely 创建线串几何图形的类。Class for creating line string geometries from shapely.
        Polygon: 从 shapely 创建多边形几何图形的类。Class for creating polygon geometries from shapely.
        Point: 从 shapely 创建点几何图形的类。Class for creating point geometries from shapely.
        prep: 来自 shapely 的预处理几何函数，用于优化空间操作。Prepared geometry function from shapely for optimized spatial operations.
        CFG (dict[str, Any]): 从 YAML 文件加载并通过 kwargs 更新的配置字典。Configuration dictionary loaded from YAML file and updated with kwargs.
        LOGGER: 用于解决方案特定日志记录的日志记录器实例。Logger instance for solution-specific logging.
        annotator: 用于在图像上绘制的标注器实例。Annotator instance for drawing on images.
        tracks: 来自最新推理的 YOLO 跟踪结果。YOLO tracking results from the latest inference.
        track_data: 从跟踪结果中提取的跟踪数据（边界框或 OBB）。Extracted tracking data (boxes or OBB) from tracks.
        boxes (list): 跟踪结果中的边界框坐标。Bounding box coordinates from tracking results.
        clss (list[int]): 跟踪结果中的类别索引。Class indices from tracking results.
        track_ids (list[int]): 跟踪结果中的跟踪 ID。Track IDs from tracking results.
        confs (list[float]): 跟踪结果中的置信度分数。Confidence scores from tracking results.
        track_line: 用于存储跟踪历史的当前跟踪线。Current track line for storing tracking history.
        masks: 跟踪结果中的分割掩码。Segmentation masks from tracking results.
        r_s: 用于空间操作的区域或线几何对象。Region or line geometry object for spatial operations.
        frame_no (int): 用于日志记录的当前帧编号。Current frame number for logging purposes.
        region (list[tuple[int, int]]): 定义感兴趣区域的坐标元组列表。List of coordinate tuples defining region of interest.
        line_width (int): 可视化中使用的线条宽度。Width of lines used in visualizations.
        model (YOLO): 已加载的 YOLO 模型实例。Loaded YOLO model instance.
        names (dict[int, str]): 将类别索引映射到类别名称的字典。Dictionary mapping class indices to class names.
        classes (list[int]): 要跟踪的类别索引列表。List of class indices to track.
        show_conf (bool): 在标注中显示置信度分数的标志。Flag to show confidence scores in annotations.
        show_labels (bool): 在标注中显示类别标签的标志。Flag to show class labels in annotations.
        device (str): 模型推理的设备。Device for model inference.
        track_add_args (dict[str, Any]): 跟踪配置的附加参数。Additional arguments for tracking configuration.
        env_check (bool): 指示环境是否支持图像显示的标志。Flag indicating whether environment supports image display.
        track_history (defaultdict): 存储每个对象跟踪历史的字典。Dictionary storing tracking history for each object.
        profilers (tuple): 用于性能监控的分析器实例。Profiler instances for performance monitoring.

    Methods:
        adjust_box_label: 为边界框生成格式化标签。Generate formatted label for bounding box.
        extract_tracks: 应用目标跟踪并从输入图像中提取轨迹。Apply object tracking and extract tracks from input image.
        store_tracking_history: 存储给定跟踪 ID 和边界框的目标跟踪历史。Store object tracking history for given track ID and bounding box.
        initialize_region: 根据配置初始化计数区域和线段。Initialize counting region and line segment based on configuration.
        display_output: 显示处理结果，包括帧或保存的结果。Display processing results including frames or saved results.
        process: 由每个解决方案子类实现的处理方法。Process method to be implemented by each Solution subclass.

    Examples:
        >>> solution = BaseSolution(model="yolo11n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        """
        使用配置设置和 YOLO 模型初始化 BaseSolution 类。
        Initialize the BaseSolution class with configuration settings and YOLO model.

        Args:
            is_cli (bool): 如果设置为 True，则启用 CLI 模式。Enable CLI mode if set to True.
            **kwargs (Any): 覆盖默认值的附加配置参数。Additional configuration parameters that override defaults.
        """
        # 从配置对象中获取所有配置参数并存储为字典
        self.CFG = vars(SolutionConfig().update(**kwargs))
        # 存储日志记录器对象，供多个解决方案类使用
        self.LOGGER = LOGGER  # Store logger object to be used in multiple solution classes

        # 检查并确保 shapely 库版本 >= 2.0.0
        check_requirements("shapely>=2.0.0")
        # 导入 shapely 几何类用于空间操作
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        # 存储几何类用于后续创建空间对象
        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        # 初始化标注器（稍后会为每一帧重新创建）
        self.annotator = None  # Initialize annotator
        # 存储最新的跟踪结果
        self.tracks = None
        # 存储提取的跟踪数据（边界框或 OBB）
        self.track_data = None
        # 初始化边界框、类别、跟踪 ID 等列表
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        # 存储区域或线的几何对象
        self.r_s = None
        # 帧编号，用于日志记录（从 -1 开始，处理时递增）
        self.frame_no = -1  # Only for logging

        # 记录解决方案配置信息
        self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")
        # 存储区域数据供其他类使用
        self.region = self.CFG["region"]  # Store region data for other classes usage
        # 存储线条宽度用于绘制
        self.line_width = self.CFG["line_width"]

        # 加载模型并存储附加信息（类别、显示置信度、显示标签）
        # Load Model and store additional information (classes, show_conf, show_label)
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo11n.pt"
        # 初始化 YOLO 模型
        self.model = YOLO(self.CFG["model"])
        # 获取类别名称映射
        self.names = self.model.names
        # 获取要跟踪的类别列表
        self.classes = self.CFG["classes"]
        # 是否显示置信度
        self.show_conf = self.CFG["show_conf"]
        # 是否显示标签
        self.show_labels = self.CFG["show_labels"]
        # 推理设备（CPU/GPU）
        self.device = self.CFG["device"]

        # 跟踪器的附加参数，用于高级配置
        # Tracker additional arguments for advance configuration
        self.track_add_args = {
            k: self.CFG[k] for k in {"iou", "conf", "device", "max_det", "half", "tracker"}
        }  # verbose must be passed to track method; setting it False in YOLO still logs the track information.

        # 如果是 CLI 模式且未提供源视频，则下载默认演示视频
        if is_cli and self.CFG["source"] is None:
            # 根据模型类型选择合适的演示视频
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            self.LOGGER.warning(f"source not provided. using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            # 从 ultralytics 资源库下载演示视频
            safe_download(f"{ASSETS_URL}/{d_s}")  # download source from ultralytics assets
            # 设置默认源
            self.CFG["source"] = d_s  # set default source

        # 初始化环境和区域设置
        # Initialize environment and region setup
        # 检查环境是否支持图像显示
        self.env_check = check_imshow(warn=True)
        # 初始化跟踪历史字典，用于存储每个对象的轨迹
        self.track_history = defaultdict(list)

        # 初始化性能分析器，用于监控跟踪和解决方案处理的耗时
        self.profilers = (
            ops.Profile(device=self.device),  # track - 跟踪耗时
            ops.Profile(device=self.device),  # solution - 解决方案处理耗时
        )

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        """
        为边界框生成格式化标签

        该方法使用类别索引和置信度分数构建边界框的标签字符串。如果提供了追踪ID，
        可选择将其包含在内。标签格式根据self.show_conf和self.show_labels中定义的显示设置进行调整。

        Args:
            cls (int): 检测到的目标的类别索引
            conf (float): 检测的置信度分数
            track_id (int, optional): 追踪目标的唯一标识符

        Returns:
            (str | None): 如果self.show_labels为True，返回格式化的标签字符串；否则返回None
        """
        name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None

    def extract_tracks(self, im0: np.ndarray) -> None:
        """
        应用目标追踪并从输入图像或帧中提取轨迹

        Args:
            im0 (np.ndarray): 输入图像或帧

        Examples:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        is_obb = self.tracks.obb is not None
        self.track_data = self.tracks.obb if is_obb else self.tracks.boxes  # Extract tracks for OBB or object detection

        if self.track_data and self.track_data.is_track:
            self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.LOGGER.warning("No tracks found.")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    def store_tracking_history(self, track_id: int, box) -> None:
        """
        存储目标的追踪历史

        该方法通过将边界框的中心点附加到追踪线来更新给定目标的追踪历史。
        它在追踪历史中最多维护30个点。

        Args:
            track_id (int): 追踪目标的唯一标识符
            box (list[float]): 目标的边界框坐标，格式为[x1, y1, x2, y2]

        Examples:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # Store tracking history
        self.track_line = self.track_history[track_id]
        self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self) -> None:
        """
        根据配置设置初始化计数区域和线段

        如果未指定区域，使用默认的矩形区域坐标。根据区域点数判断是多边形区域还是线段。
        """
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # region or line

    def display_output(self, plot_im: np.ndarray) -> None:
        """
        显示处理结果，包括显示帧、打印计数或保存结果

        该方法负责可视化目标检测和追踪过程的输出。它显示标注后的处理帧，并允许用户交互关闭显示。

        Args:
            plot_im (np.ndarray): 已处理和标注的图像或帧

        使用示例:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        注意事项:
            - 仅当'show'配置设置为True且环境支持图像显示时才会显示输出
            - 按'q'键可以关闭显示
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # Closes current frame window
                return

    def process(self, *args: Any, **kwargs: Any):
        """处理方法应由每个Solution子类实现"""

    def __call__(self, *args: Any, **kwargs: Any):
        """
        允许实例像函数一样被调用，支持灵活的参数

        该方法使Solution对象可调用，自动执行子类的process方法并记录性能指标。
        """
        with self.profilers[1]:
            result = self.process(*args, **kwargs)  # Call the subclass-specific process method
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # solution time = process - track
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)  # Only for logging.
            LOGGER.info(
                f"{self.frame_no}: {result.plot_im.shape[0]}x{result.plot_im.shape[1]} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape "
                f"(1, {getattr(self.model, 'ch', 3)}, {result.plot_im.shape[0]}, {result.plot_im.shape[1]})\n"
            )
        return result


class SolutionAnnotator(Annotator):
    """
    解决方案标注器(SolutionAnnotator)类：用于可视化和分析计算机视觉任务的专用标注器

    该类扩展了基础Annotator类，为Ultralytics解决方案提供了绘制区域、质心、追踪轨迹和视觉标注的额外方法。
    它为目标检测、追踪、姿态估计和数据分析等各种计算机视觉应用提供全面的可视化能力。

    核心功能：
    1. 绘制计数区域和线段
    2. 显示队列计数和分析统计
    3. 姿态角度估计和可视化
    4. 关键点绘制和连接
    5. 距离测量和标注
    6. 物体标签和追踪可视化

    属性:
        im (np.ndarray): 被标注的图像
        line_width (int): 标注中使用的线条粗细
        font_size (int): 文本标注的字体大小
        font (str): 用于文本渲染的字体文件路径
        pil (bool): 是否使用PIL进行文本渲染
        example (str): 用于检测非ASCII标签以使用PIL渲染的示例文本

    方法:
        draw_region: 使用指定的点、颜色和粗细绘制区域
        queue_counts_display: 在指定区域显示队列计数
        display_analytics: 显示停车场管理的整体统计信息
        estimate_pose_angle: 计算物体姿态中三个点之间的角度
        draw_specific_kpts: 在图像上绘制特定关键点
        plot_workout_information: 在图像上绘制标记文本框
        plot_angle_and_count_and_stage: 可视化运动监测的角度、步数和阶段
        plot_distance_and_line: 显示质心之间的距离并用线连接它们
        display_objects_labels: 用物体类别标签标注边界框
        sweep_annotator: 可视化垂直扫描线和可选标签
        visioneye: 将物体质心映射并连接到视觉"眼睛"点
        adaptive_label: 在边界框中心绘制圆形或矩形背景形状标签

    使用示例:
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """
        使用图像初始化SolutionAnnotator类用于标注

        Args:
            im (np.ndarray): 要标注的图像
            line_width (int, optional): 在图像上绘制的线条粗细
            font_size (int, optional): 文本标注的字体大小
            font (str): 字体文件路径
            pil (bool): 是否使用PIL渲染文本
            example (str): 用于检测非ASCII标签以使用PIL渲染的示例文本
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(
        self,
        reg_pts: list[tuple[int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5,
    ):
        """
        在图像上绘制区域或线段

        Args:
            reg_pts (list[tuple[int, int]], optional): 区域点（线段2个点，区域4个以上点）
            color (tuple[int, int, int]): 区域的BGR颜色值（OpenCV格式）
            thickness (int): 绘制区域的线条粗细
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def queue_counts_display(
        self,
        label: str,
        points: list[tuple[int, int]] | None = None,
        region_color: tuple[int, int, int] = (255, 255, 255),
        txt_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """
        在图像上显示队列计数，文本居中于点位置，可自定义字体大小和颜色

        Args:
            label (str): 队列计数标签
            points (list[tuple[int, int]], optional): 用于计算中心点以显示文本的区域点
            region_color (tuple[int, int, int]): BGR队列区域颜色（OpenCV格式）
            txt_color (tuple[int, int, int]): BGR文本颜色（OpenCV格式）
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
    ):
        """
        显示解决方案的整体统计信息（如停车管理和物体计数）

        Args:
            im0 (np.ndarray): 推理图像
            text (dict[str, Any]): 标签字典
            txt_color (tuple[int, int, int]): 文本颜色（BGR，OpenCV格式）
            bg_color (tuple[int, int, int]): 背景颜色（BGR，OpenCV格式）
            margin (int): 文本与矩形之间的间隙，以获得更好的显示效果
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        """将类关键点对象转换为浮点数(x, y)元组"""
        if hasattr(point, "detach"):  # torch.Tensor
            point = point.detach()
        if hasattr(point, "cpu"):  # torch.Tensor
            point = point.cpu()
        if hasattr(point, "numpy"):  # torch.Tensor
            point = point.numpy()
        if hasattr(point, "tolist"):  # numpy / torch
            point = point.tolist()
        return float(point[0]), float(point[1])

    @staticmethod
    @lru_cache(maxsize=256)
    def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """计算用于运动监测的三点之间角度（带缓存）"""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle if angle <= 180.0 else (360 - angle)

    @staticmethod
    def estimate_pose_angle(a: Any, b: Any, c: Any) -> float:
        """
        计算用于运动监测的三点之间角度

        Args:
            a (Any): 第一个点的坐标（例如list/tuple/NumPy数组/torch张量）
            b (Any): 第二个点（顶点）的坐标
            c (Any): 第三个点的坐标

        Returns:
            (float): 三点之间的角度（度数）
        """
        a_xy, b_xy, c_xy = (
            SolutionAnnotator._point_xy(a),
            SolutionAnnotator._point_xy(b),
            SolutionAnnotator._point_xy(c),
        )
        return SolutionAnnotator._estimate_pose_angle_cached(a_xy, b_xy, c_xy)

    def draw_specific_kpts(
        self,
        keypoints: list[list[float]],
        indices: list[int] | None = None,
        radius: int = 2,
        conf_thresh: float = 0.25,
    ) -> np.ndarray:
        """
        绘制用于健身步数计数的特定关键点

        Args:
            keypoints (list[list[float]]): 要绘制的关键点数据，每个格式为[x, y, confidence]
            indices (list[int], optional): 要绘制的关键点索引
            radius (int): 关键点半径
            conf_thresh (float): 关键点置信度阈值

        Returns:
            (np.ndarray): 绘制了关键点的图像

        注意事项:
            关键点格式：[x, y]或[x, y, confidence]
            原地修改self.im
        """
        indices = indices or [2, 5, 7]
        points = [(int(k[0]), int(k[1])) for i, k in enumerate(keypoints) if i in indices and k[2] >= conf_thresh]

        # Draw lines between consecutive points
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Draw circles for keypoints
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(
        self,
        display_text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> int:
        """
        在图像上绘制带背景的运动文本

        Args:
            display_text (str): 要显示的文本
            position (tuple[int, int]): 文本在图像上的放置坐标(x, y)
            color (tuple[int, int, int]): 文本背景颜色
            txt_color (tuple[int, int, int]): 文本前景颜色

        Returns:
            (int): 文本高度
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

        # Draw background rectangle
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # Draw text
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self,
        angle_text: str,
        count_text: str,
        stage_text: str,
        center_kpt: list[int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """
        绘制用于运动监测的姿态角度、计数值和阶段

        Args:
            angle_text (str): 运动监测的角度值
            count_text (str): 运动监测的计数值
            stage_text (str): 运动监测的阶段判断
            center_kpt (list[int]): 运动监测的质心姿态索引
            color (tuple[int, int, int]): 文本背景颜色
            txt_color (tuple[int, int, int]): 文本前景颜色
        """
        # Format text
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # Draw angle, count and stage text
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(
        self,
        pixels_distance: float,
        centroids: list[tuple[int, int]],
        line_color: tuple[int, int, int] = (104, 31, 17),
        centroid_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """
        在帧上绘制两个质心之间的距离和连线

        Args:
            pixels_distance (float): 两个边界框质心之间的像素距离
            centroids (list[tuple[int, int]]): 边界框质心数据
            line_color (tuple[int, int, int]): 距离线颜色
            centroid_color (tuple[int, int, int]): 边界框质心颜色
        """
        # Get the text size
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # Define corners with 10-pixel margin and draw rectangle
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # Calculate the position for the text with a 10-pixel margin and draw text
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(
        self,
        im0: np.ndarray,
        text: str,
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        x_center: float,
        y_center: float,
        margin: int,
    ):
        """
        在停车管理应用中显示边界框标签

        Args:
            im0 (np.ndarray): 推理图像
            text (str): 物体/类别名称
            txt_color (tuple[int, int, int]): 文本前景显示颜色
            bg_color (tuple[int, int, int]): 文本背景显示颜色
            x_center (float): 边界框的x位置中心点
            y_center (float): 边界框的y位置中心点
            margin (int): 文本与矩形之间的间隙，以获得更好的显示效果
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),  # Ensure color values are int
            -1,
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),  # Ensure color values are int
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def sweep_annotator(
        self,
        line_x: int = 0,
        line_y: int = 0,
        label: str | None = None,
        color: tuple[int, int, int] = (221, 0, 186),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """
        绘制扫描标注线和可选标签

        Args:
            line_x (int): 扫描线的x坐标
            line_y (int): 扫描线的y坐标限制
            label (str, optional): 在扫描线中心绘制的文本标签。如果为None，则不绘制标签
            color (tuple[int, int, int]): 线条和标签背景的BGR颜色（OpenCV格式）
            txt_color (tuple[int, int, int]): 标签文本的BGR颜色（OpenCV格式）
        """
        # Draw the sweep line
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # Draw label, if provided
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def visioneye(
        self,
        box: list[float],
        center_point: tuple[int, int],
        color: tuple[int, int, int] = (235, 219, 11),
        pin_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """
        执行精确的人类视觉眼睛映射和绘制

        Args:
            box (list[float]): 边界框坐标，格式为[x1, y1, x2, y2]
            center_point (tuple[int, int]): 视觉眼睛视图的中心点
            color (tuple[int, int, int]): 物体质心和线条颜色
            pin_color (tuple[int, int, int]): 视觉眼睛点颜色
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def adaptive_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        shape: str = "rect",
        margin: int = 5,
    ):
        """
        在给定边界框内居中绘制带背景矩形或圆形的标签

        Args:
            box (tuple[float, float, float, float]): 边界框坐标(x1, y1, x2, y2)
            label (str): 要显示的文本标签
            color (tuple[int, int, int]): 矩形的背景颜色(B, G, R)
            txt_color (tuple[int, int, int]): 文本颜色(B, G, R)
            shape (str): 标签形状。选项："circle"或"rect"
            margin (int): 文本与矩形边框之间的边距
        """
        if shape == "circle" and len(label) > 3:
            LOGGER.warning(f"Length of label is {len(label)}, only first 3 letters will be used for circle annotation.")
            label = label[:3]

        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # Bounding-box center
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]  # Get size of the text
        text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2  # Calculate top-left corner of text

        if shape == "circle":
            cv2.circle(
                self.im,
                (x_center, y_center),
                int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,  # Calculate the radius
                color,
                -1,
            )
        else:
            cv2.rectangle(
                self.im,
                (text_x - margin, text_y - text_size[1] - margin),  # Calculate coordinates of the rectangle
                (text_x + text_size[0] + margin, text_y + margin),  # Calculate coordinates of the rectangle
                color,
                -1,
            )

        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),  # Calculate top-left corner of the text
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )


class SolutionResults:
    """
    解决方案结果(SolutionResults)类：封装Ultralytics解决方案的结果

    该类用于存储和管理解决方案流程生成的各种输出，包括计数、角度、运动阶段和其他分析数据。
    它为不同的计算机视觉解决方案（如物体计数、姿态估计和追踪分析）提供结构化的结果访问和操作方式。

    属性:
        plot_im (np.ndarray): 处理后的图像，包含计数、模糊或解决方案的其他效果
        in_count (int): 视频流中"进入"的总计数
        out_count (int): 视频流中"离开"的总计数
        classwise_count (dict[str, int]): 包含按类别分类的物体计数的字典
        queue_count (int): 队列或等待区域中的物体计数
        workout_count (int): 运动重复次数计数
        workout_angle (float): 运动练习期间计算的角度
        workout_stage (str): 运动的当前阶段
        pixels_distance (float): 两点或物体之间计算的像素距离
        available_slots (int): 监控区域中可用的槽位数量
        filled_slots (int): 监控区域中已填充的槽位数量
        email_sent (bool): 指示是否发送了电子邮件通知的标志
        total_tracks (int): 追踪的物体总数
        region_counts (dict[str, int]): 特定区域内的物体计数
        speed_dict (dict[str, float]): 包含追踪物体速度信息的字典
        total_crop_objects (int): 使用ObjectCropper类裁剪的物体总数
        speed (dict[str, float]): 追踪和解决方案处理的性能计时信息
    """

    def __init__(self, **kwargs):
        """
        使用默认值或用户指定值初始化SolutionResults对象

        Args:
            **kwargs (Any): 覆盖默认属性值的可选参数
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = 0
        self.workout_angle = 0.0
        self.workout_stage = None
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}  # for speed estimation
        self.total_crop_objects = 0
        self.speed = {}

        # Override with user-defined values
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """
        返回SolutionResults对象的格式化字符串表示

        Returns:
            (str): 列出非空属性的字符串表示
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # Exclude `plot_im` explicitly
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
