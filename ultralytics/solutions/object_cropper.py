from pathlib import Path
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionResults
from ultralytics.utils.plotting import save_one_box


class ObjectCropper(BaseSolution):
    """
    目标裁剪器(ObjectCropper)类：管理实时视频流或图像中检测到的目标的裁剪

    该类继承自BaseSolution类，提供基于检测到的边界框裁剪目标的功能。
    裁剪的图像保存到指定目录，供进一步分析或使用。主要应用于目标提取、数据集构建、
    单独目标分析等场景。

    核心功能：
    1. 检测图像中的目标
    2. 根据边界框裁剪每个目标
    3. 将裁剪的目标保存为独立图像文件
    4. 自动管理文件命名和存储

    属性:
        crop_dir (str): 存储裁剪目标图像的目录
        crop_idx (int): 已裁剪目标的总数计数器
        iou (float): 非极大值抑制的IoU（交并比）阈值
        conf (float): 过滤检测结果的置信度阈值

    方法:
        process: 从输入图像中裁剪检测到的目标并保存到输出目录

    使用示例:
        >>> from ultralytics.solutions import ObjectCropper
        >>> cropper = ObjectCropper(crop_dir="./crops")
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = cropper.process(frame)
        >>> print(f"裁剪的目标总数: {cropper.crop_idx}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化ObjectCropper类，用于从检测到的边界框中裁剪目标

        Args:
            **kwargs (Any): 传递给父类的关键字参数和配置，包括:
                - crop_dir (str): 保存裁剪目标图像的目录路径
                - conf (float): 置信度阈值
                - iou (float): IoU阈值
                - model: YOLO模型路径
        """
        super().__init__(**kwargs)

        self.crop_dir = self.CFG["crop_dir"]  # 用于存储裁剪检测结果的目录
        Path(self.crop_dir).mkdir(parents=True, exist_ok=True)
        if self.CFG["show"]:
            self.LOGGER.warning(f"ObjectCropper不支持show=True；将裁剪结果保存到 '{self.crop_dir}'。")
            self.CFG["show"] = False
        self.crop_idx = 0  # 初始化已裁剪目标总数的计数器
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"]

    def process(self, im0) -> SolutionResults:
        """
        从输入图像中裁剪检测到的目标并将其保存为独立图像

        该方法实现完整的目标裁剪流程：
        1. 使用YOLO模型检测图像中的目标
        2. 对每个检测到的边界框：
           - 增加裁剪计数器
           - 使用save_one_box函数裁剪边界框区域
           - 将裁剪的图像保存为"crop_{序号}.jpg"
        3. 返回处理结果

        Args:
            im0 (np.ndarray): 包含待检测目标的输入图像

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_crop_objects: 裁剪的目标总数

        使用示例:
            >>> cropper = ObjectCropper()
            >>> frame = cv2.imread("image.jpg")
            >>> results = cropper.process(frame)
            >>> print(f"裁剪的目标总数: {results.total_crop_objects}")
        """
        with self.profilers[0]:
            results = self.model.predict(
                im0,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                device=self.CFG["device"],
                verbose=False,
            )[0]
            self.clss = results.boxes.cls.tolist()  # 仅用于日志记录

        for box in results.boxes:
            self.crop_idx += 1
            save_one_box(
                box.xyxy,
                im0,
                file=Path(self.crop_dir) / f"crop_{self.crop_idx}.jpg",
                BGR=True,
            )

        # 返回SolutionResults对象
        return SolutionResults(plot_im=im0, total_crop_objects=self.crop_idx)
