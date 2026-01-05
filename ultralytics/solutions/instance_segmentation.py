from typing import Any

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class InstanceSegmentation(BaseSolution):
    """
    实例分割(InstanceSegmentation)类：在图像或视频流中管理实例分割

    该类继承自BaseSolution类，提供执行实例分割的功能，包括绘制分割掩码、边界框和标签。
    实例分割不仅识别目标的类别和位置，还能精确分割出每个目标实例的像素级轮廓。

    核心功能：
    1. 使用分割模型检测和分割目标实例
    2. 为每个实例生成精确的掩码
    3. 绘制边界框、类别标签和置信度
    4. 支持自定义显示选项

    属性:
        model (str): 用于推理的分割模型
        line_width (int): 边界框和文本线条的宽度
        names (dict[int, str]): 将类别索引映射到类别名称的字典
        clss (list[int]): 检测到的类别索引列表
        track_ids (list[int]): 检测到的实例的追踪ID列表
        masks (list[np.ndarray]): 检测到的实例的分割掩码列表
        show_conf (bool): 是否显示置信度分数
        show_labels (bool): 是否显示类别标签
        show_boxes (bool): 是否显示边界框

    方法:
        process: 处理输入图像以执行实例分割并标注结果
        extract_tracks: 从模型预测中提取轨迹，包括边界框、类别和掩码

    使用示例:
        >>> from ultralytics.solutions import InstanceSegmentation
        >>> segmenter = InstanceSegmentation(model="yolo11n-seg.pt")
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.process(frame)
        >>> print(f"分割的实例总数: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化InstanceSegmentation类，用于检测和标注分割实例

        Args:
            **kwargs (Any): 传递给BaseSolution父类的关键字参数，包括:
                - model (str): 模型名称或路径，默认为 "yolo11n-seg.pt"
                - show_conf (bool): 是否显示置信度
                - show_labels (bool): 是否显示标签
                - show_boxes (bool): 是否显示边界框
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-seg.pt")
        super().__init__(**kwargs)

        self.show_conf = self.CFG.get("show_conf", True)
        self.show_labels = self.CFG.get("show_labels", True)
        self.show_boxes = self.CFG.get("show_boxes", True)

    def process(self, im0) -> SolutionResults:
        """
        对输入图像执行实例分割并标注结果

        该方法实现完整的实例分割流程：
        1. 提取追踪轨迹（边界框、类别和掩码）
        2. 获取分割掩码数据
        3. 如果检测到掩码：
           - 创建Results对象包含所有检测信息
           - 使用plot方法绘制分割结果，包括：
             - 彩色分割掩码
             - 边界框（可选）
             - 类别标签（可选）
             - 置信度分数（可选）
        4. 如果未检测到掩码，记录警告并返回原图
        5. 显示并返回标注结果

        Args:
            im0 (np.ndarray): 待分割的输入图像

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 标注后的图像
                - total_tracks: 追踪的实例总数

        使用示例:
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.process(frame)
            >>> print(f"检测到 {summary.total_tracks} 个实例")
        """
        self.extract_tracks(im0)  # 提取轨迹（边界框、类别和掩码）
        self.masks = getattr(self.tracks, "masks", None)

        # 遍历检测到的类别、追踪ID和分割掩码
        if self.masks is None:
            self.LOGGER.warning("未检测到掩码！请确保您使用的是支持的Ultralytics分割模型。")
            plot_im = im0
        else:
            results = Results(im0, path=None, names=self.names, boxes=self.track_data.data, masks=self.masks.data)
            plot_im = results.plot(
                line_width=self.line_width,
                boxes=self.show_boxes,
                conf=self.show_conf,
                labels=self.show_labels,
                color_mode="instance",
            )

        self.display_output(plot_im)  # 使用基类函数显示标注输出

        # 返回SolutionResults对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
