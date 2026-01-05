import numpy as np  # 导入NumPy库，用于数组和数值计算
import torch  # 导入PyTorch深度学习框架

from ultralytics.data.augment import LoadVisualPrompt  # 导入视觉提示加载器，用于加载和处理视觉提示数据
from ultralytics.models.yolo.detect import DetectionPredictor  # 导入YOLO目标检测预测器基类
from ultralytics.models.yolo.segment import SegmentationPredictor  # 导入YOLO分割预测器基类


class YOLOEVPDetectPredictor(DetectionPredictor):
    """
    YOLO-EVP (Enhanced Visual Prompting) 目标检测预测器混入类。
    A mixin class for YOLO-EVP (Enhanced Visual Prompting) predictors.

    该混入类为使用视觉提示的YOLO模型提供通用功能，包括模型设置、提示处理和预处理转换。
    This mixin provides common functionality for YOLO models that use visual prompting, including model setup, prompt
    handling, and preprocessing transformations.

    属性 Attributes:
        model (torch.nn.Module): 用于推理的YOLO模型。The YOLO model for inference.
        device (torch.device): 运行模型的设备（CPU或CUDA）。Device to run the model on (CPU or CUDA).
        prompts (dict | torch.Tensor): 包含类别索引和边界框或掩码的视觉提示。
            Visual prompts containing class indices and bounding boxes or masks.

    方法 Methods:
        setup_model: 初始化YOLO模型并设置为评估模式。Initialize the YOLO model and set it to evaluation mode.
        set_prompts: 为模型设置视觉提示。Set the visual prompts for the model.
        pre_transform: 在推理前预处理图像和提示。Preprocess images and prompts before inference.
        inference: 使用视觉提示运行推理。Run inference with visual prompts.
        get_vpe: 处理源数据以获取视觉提示嵌入。Process source to get visual prompt embeddings.
    """

    def setup_model(self, model, verbose: bool = True):
        """
        设置用于预测的模型。
        Set up the model for prediction.

        参数 Args:
            model (torch.nn.Module): 要加载或使用的模型。Model to load or use.
            verbose (bool, optional): 如果为True，提供详细的日志输出。If True, provides detailed logging.
        """
        super().setup_model(model, verbose=verbose)
        self.done_warmup = True  # 标记预热已完成，避免重复预热操作

    def set_prompts(self, prompts):
        """
        为模型设置视觉提示。
        Set the visual prompts for the model.

        参数 Args:
            prompts (dict): 包含类别索引和边界框或掩码的字典。必须包含带有类别索引的'cls'键。
                Dictionary containing class indices and bounding boxes or masks. Must include a 'cls' key
                with class indices.
        """
        self.prompts = prompts

    def pre_transform(self, im):
        """
        在推理前预处理图像和提示。
        Preprocess images and prompts before inference.

        该方法对输入图像应用letterbox处理，并相应地转换视觉提示（边界框或掩码）。
        This method applies letterboxing to the input image and transforms the visual prompts (bounding boxes or masks)
        accordingly.

        参数 Args:
            im (list): 包含单个输入图像的列表。List containing a single input image.

        返回 Returns:
            (list): 准备好用于模型推理的预处理图像。Preprocessed image ready for model inference.

        异常 Raises:
            ValueError: 如果提示中既未提供有效的边界框也未提供掩码。
                If neither valid bounding boxes nor masks are provided in the prompts.
        """
        img = super().pre_transform(im)
        bboxes = self.prompts.pop("bboxes", None)  # 从提示中提取边界框
        masks = self.prompts.pop("masks", None)  # 从提示中提取掩码
        category = self.prompts["cls"]  # 获取类别信息
        if len(img) == 1:
            # 处理单张图像的情况
            visuals = self._process_single_image(img[0].shape[:2], im[0].shape[:2], category, bboxes, masks)
            prompts = visuals.unsqueeze(0).to(self.device)  # (1, N, H, W) 添加批次维度并转移到设备
        else:
            # 处理批量图像的情况
            # 注意：目前仅支持边界框作为提示
            assert bboxes is not None, f"Expected bboxes, but got {bboxes}!"
            # 注意：需要list[np.ndarray]格式
            assert isinstance(bboxes, list) and all(isinstance(b, np.ndarray) for b in bboxes), (
                f"Expected list[np.ndarray], but got {bboxes}!"
            )
            assert isinstance(category, list) and all(isinstance(b, np.ndarray) for b in category), (
                f"Expected list[np.ndarray], but got {category}!"
            )
            assert len(im) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(im)}vs{len(category)}vs{len(bboxes)}!"
            )
            # 对批次中的每张图像进行处理
            visuals = [
                self._process_single_image(img[i].shape[:2], im[i].shape[:2], category[i], bboxes[i])
                for i in range(len(img))
            ]
            # 对不同长度的序列进行填充，使其具有相同长度
            prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)  # (B, N, H, W)
        # 根据模型精度设置提示的数据类型
        self.prompts = prompts.half() if self.model.fp16 else prompts.float()
        return img

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        """
        通过调整边界框或掩码的大小并生成视觉表示来处理单张图像。
        Process a single image by resizing bounding boxes or masks and generating visuals.

        参数 Args:
            dst_shape (tuple): 图像的目标形状（高度，宽度）。The target shape (height, width) of the image.
            src_shape (tuple): 图像的原始形状（高度，宽度）。The original shape (height, width) of the image.
            category (str): 用于视觉提示的图像类别。The category of the image for visual prompts.
            bboxes (list | np.ndarray, optional): 格式为[x1, y1, x2, y2]的边界框列表。
                A list of bounding boxes in the format [x1, y1, x2, y2].
            masks (np.ndarray, optional): 与图像对应的掩码列表。A list of masks corresponding to the image.

        返回 Returns:
            (torch.Tensor): 图像的处理后的视觉表示。The processed visuals for the image.

        异常 Raises:
            ValueError: 如果既未提供`bboxes`也未提供`masks`。If neither `bboxes` nor `masks` are provided.
        """
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]  # 确保边界框是二维数组
            # 计算缩放因子并调整边界框
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # gain = old / new
            bboxes *= gain  # 应用缩放
            # 调整x坐标以考虑letterbox填充
            bboxes[..., 0::2] += round((dst_shape[1] - src_shape[1] * gain) / 2 - 0.1)
            # 调整y坐标以考虑letterbox填充
            bboxes[..., 1::2] += round((dst_shape[0] - src_shape[0] * gain) / 2 - 0.1)
        elif masks is not None:
            # 调整掩码大小并处理
            resized_masks = super().pre_transform(masks)
            masks = np.stack(resized_masks)  # (N, H, W) 堆叠为三维数组
            masks[masks == 114] = 0  # 将填充值重置为0（114是letterbox的默认填充值）
        else:
            raise ValueError("Please provide valid bboxes or masks")

        # 使用视觉提示加载器生成视觉表示
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)

    def inference(self, im, *args, **kwargs):
        """
        使用视觉提示运行推理。
        Run inference with visual prompts.

        参数 Args:
            im (torch.Tensor): 输入图像张量。Input image tensor.
            *args (Any): 可变长度参数列表。Variable length argument list.
            **kwargs (Any): 任意关键字参数。Arbitrary keyword arguments.

        返回 Returns:
            (torch.Tensor): 模型预测结果。Model prediction results.
        """
        return super().inference(im, vpe=self.prompts, *args, **kwargs)

    def get_vpe(self, source):
        """
        处理源数据以获取视觉提示嵌入（VPE）。
        Process the source to get the visual prompt embeddings (VPE).

        参数 Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple):
                要进行预测的图像源。接受多种类型，包括文件路径、URL、PIL图像、numpy数组和torch张量。
                The source of the image to make predictions on. Accepts various types including file paths, URLs,
                PIL images, numpy arrays, and torch tensors.

        返回 Returns:
            (torch.Tensor): 来自模型的视觉提示嵌入（VPE）。The visual prompt embeddings (VPE) from the model.
        """
        self.setup_source(source)
        assert len(self.dataset) == 1, "get_vpe only supports one image!"
        for _, im0s, _ in self.dataset:
            im = self.preprocess(im0s)
            return self.model(im, vpe=self.prompts, return_vpe=True)


class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor):
    """
    YOLO-EVP分割任务预测器，结合了检测和分割能力。
    Predictor for YOLO-EVP segmentation tasks combining detection and segmentation capabilities.

    该类继承了YOLOEVPDetectPredictor的视觉提示处理能力和SegmentationPredictor的分割功能，
    提供完整的实例分割预测能力。
    This class inherits visual prompt processing capabilities from YOLOEVPDetectPredictor and segmentation
    functionality from SegmentationPredictor, providing complete instance segmentation prediction capabilities.
    """

    pass
