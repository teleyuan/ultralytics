from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    """
    目标模糊器(ObjectBlurrer)类：管理实时视频流中检测到的目标的模糊处理

    该类继承自BaseSolution类，提供基于检测到的边界框对目标进行模糊处理的功能。
    模糊区域直接在输入图像上更新，用于隐私保护或其他效果。主要应用于视频监控中的隐私保护、
    敏感信息遮挡等场景。

    核心功能：
    1. 检测图像中的目标
    2. 对检测到的目标区域应用模糊效果
    3. 可调节模糊强度
    4. 标注处理后的结果

    属性:
        blur_ratio (int): 应用于检测目标的模糊效果强度（值越高模糊越强）
        iou (float): 目标检测的交并比阈值
        conf (float): 目标检测的置信度阈值

    方法:
        process: 对输入图像中检测到的目标应用模糊效果
        extract_tracks: 从检测到的目标中提取追踪信息
        display_output: 显示处理后的输出图像

    使用示例:
        >>> from ultralytics.solutions import ObjectBlurrer
        >>> blurrer = ObjectBlurrer(blur_ratio=0.7)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = blurrer.process(frame)
        >>> print(f"模糊处理的目标总数: {processed_results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化ObjectBlurrer类，用于对视频流或图像中检测到的目标应用模糊效果

        Args:
            **kwargs (Any): 传递给父类的关键字参数和配置，包括:
                - blur_ratio (float): 模糊效果强度（0.1-1.0，默认0.5）
                - model: YOLO模型路径
                - conf: 置信度阈值
        """
        super().__init__(**kwargs)
        blur_ratio = self.CFG["blur_ratio"]
        if blur_ratio < 0.1:
            LOGGER.warning("模糊比例不能小于0.1，将其更新为默认值0.5")
            blur_ratio = 0.5
        self.blur_ratio = int(blur_ratio * 100)

    def process(self, im0) -> SolutionResults:
        """
        对输入图像中检测到的目标应用模糊效果

        该方法实现完整的目标模糊流程：
        1. 提取追踪信息，检测图像中的目标
        2. 对每个检测到的目标：
           - 提取边界框对应的图像区域
           - 使用cv2.blur应用模糊效果（核大小由blur_ratio决定）
           - 将模糊后的区域替换回原图像
        3. 标注边界框和标签
        4. 返回处理结果

        Args:
            im0 (np.ndarray): 包含待检测目标的输入图像

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 带有模糊目标的标注输出图像
                - total_tracks: 帧中追踪的目标总数

        使用示例:
            >>> blurrer = ObjectBlurrer()
            >>> frame = cv2.imread("image.jpg")
            >>> results = blurrer.process(frame)
            >>> print(f"模糊处理了 {results.total_tracks} 个目标")
        """
        self.extract_tracks(im0)  # 提取追踪轨迹
        annotator = SolutionAnnotator(im0, self.line_width)

        # 遍历边界框和类别
        for box, cls, conf in zip(self.boxes, self.clss, self.confs):
            # 裁剪并模糊检测到的目标
            blur_obj = cv2.blur(
                im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                (self.blur_ratio, self.blur_ratio),
            )
            # 在原始图像中更新模糊区域
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf), color=colors(cls, True)
            )  # 标注边界框

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
