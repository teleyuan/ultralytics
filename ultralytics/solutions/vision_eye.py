from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class VisionEye(BaseSolution):
    """
    视觉眼(VisionEye)类：在图像或视频流中管理目标检测和视觉映射

    该类继承自BaseSolution类，提供目标检测、视觉点映射以及用边界框和标签标注结果的功能。
    主要用于从指定视点追踪和可视化目标运动轨迹。

    属性:
        vision_point (tuple[int, int]): 视觉点坐标(x, y)，系统从该点观察目标并绘制轨迹

    方法:
        process: 处理输入图像，检测目标，标注并应用视觉映射

    使用示例:
        >>> vision_eye = VisionEye()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = vision_eye.process(frame)
        >>> print(f"检测到的实例总数: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化VisionEye类，用于目标检测和视觉映射

        Args:
            **kwargs (Any): 传递给父类的关键字参数，用于配置vision_point等参数
        """
        super().__init__(**kwargs)
        # 设置视觉点：系统从该点观察目标并绘制轨迹
        self.vision_point = self.CFG["vision_point"]

    def process(self, im0) -> SolutionResults:
        """
        对输入图像执行目标检测、视觉映射和标注

        该方法实现了完整的视觉眼处理流程：
        1. 提取目标轨迹（边界框、类别和掩码）
        2. 创建标注器并为每个目标绘制边界框和标签
        3. 从视觉点绘制到每个目标的连线，形成视觉映射效果
        4. 显示并返回标注结果

        Args:
            im0 (np.ndarray): 用于检测和标注的输入图像

        Returns:
            (SolutionResults): 包含标注图像和追踪统计信息的对象
                - plot_im: 带有边界框和视觉映射的标注输出图像
                - total_tracks: 帧中追踪的目标数量

        使用示例:
            >>> vision_eye = VisionEye()
            >>> frame = cv2.imread("image.jpg")
            >>> results = vision_eye.process(frame)
            >>> print(f"检测到 {results.total_tracks} 个目标")
        """
        self.extract_tracks(im0)  # 提取轨迹（边界框、类别和掩码）
        annotator = SolutionAnnotator(im0, self.line_width)

        for cls, t_id, box, conf in zip(self.clss, self.track_ids, self.boxes, self.confs):
            # 用边界框、标签和视觉映射标注图像
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, t_id), color=colors(int(t_id), True))
            annotator.visioneye(box, self.vision_point)

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示标注输出

        # 返回包含标注图像和追踪统计信息的SolutionResults对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
