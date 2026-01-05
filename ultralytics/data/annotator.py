"""
自动标注模块

该模块提供了使用 YOLO 和 SAM 模型自动标注图像的功能。
它结合了目标检测和实例分割，能够为图像数据集自动生成高质量的分割标注。

主要功能:
    - 使用 YOLO 模型进行目标检测
    - 使用 SAM (Segment Anything Model) 模型生成精确的分割掩码
    - 自动保存 YOLO 格式的标注文件
    - 支持批量处理图像目录

典型应用场景:
    - 快速为新数据集生成训练标注
    - 半自动标注工作流
    - 数据增强和预标注
"""

from __future__ import annotations  # 启用延迟类型注解评估，支持 Python 3.9+ 的新式类型提示

from pathlib import Path  # 用于跨平台的路径操作

# 导入 YOLO 目标检测模型和 SAM 分割模型
from ultralytics import SAM, YOLO


def auto_annotate(
    data: str | Path,
    det_model: str = "yolo11x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    使用 YOLO 目标检测模型和 SAM 分割模型自动标注图像

    该函数实现了一个两阶段的自动标注流程:
    1. 使用 YOLO 模型检测图像中的目标并获取边界框
    2. 使用 SAM 模型基于检测框生成精确的分割掩码
    3. 将结果保存为 YOLO 格式的标注文件（每行: 类别ID + 归一化的多边形坐标）

    Args:
        data (str | Path): 包含待标注图像的文件夹路径
        det_model (str): 预训练 YOLO 检测模型的路径或名称
            默认: "yolo11x.pt" (YOLO11 XLarge 模型，精度最高)
        sam_model (str): 预训练 SAM 分割模型的路径或名称
            默认: "sam_b.pt" (SAM Base 模型)
            可选: "mobile_sam.pt" (轻量级), "sam_l.pt" (大型)
        device (str): 运行模型的设备，如 'cpu', 'cuda', '0', '1' 等
            空字符串表示自动选择（优先 GPU）
        conf (float): 检测模型的置信度阈值 (0.0-1.0)
            低于此值的检测结果将被过滤，默认 0.25
        iou (float): NMS (非极大值抑制) 的 IoU 阈值 (0.0-1.0)
            用于过滤重叠的检测框，默认 0.45
        imgsz (int): 输入图像的目标尺寸（像素）
            图像将被调整为此尺寸进行推理，默认 640
        max_det (int): 每张图像最多检测的目标数量，默认 300
        classes (list[int] | None): 要保留的类别ID列表
            None 表示保留所有类别，可用于只标注特定类别的目标
        output_dir (str | Path | None): 保存标注结果的目录
            None 表示在输入目录的父目录下创建 "{输入目录名}_auto_annotate_labels"

    Returns:
        None: 函数无返回值，结果直接保存到文件系统

    Examples:
        基本用法 - 自动标注图像文件夹:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")

        只标注特定类别（如人和车）:
        >>> auto_annotate(data="my_images", classes=[0, 2], output_dir="annotations")

        使用高精度模型和自定义参数:
        >>> auto_annotate(
        ...     data="dataset/images",
        ...     det_model="yolo11x.pt",
        ...     sam_model="sam_l.pt",
        ...     conf=0.3,
        ...     device="cuda:0"
        ... )

    Note:
        - 生成的标注文件格式: 每行为 "类别ID x1 y1 x2 y2 ... xn yn"
        - 坐标已归一化到 [0, 1] 范围
        - 如果检测到的目标没有分割结果，将跳过该目标
    """
    # 加载 YOLO 检测模型
    det_model = YOLO(det_model)
    # 加载 SAM 分割模型
    sam_model = SAM(sam_model)

    # 将输入路径转换为 Path 对象，便于路径操作
    data = Path(data)

    # 如果未指定输出目录，创建默认输出目录
    if not output_dir:
        # 在输入目录的父目录下创建新文件夹，命名为 "{原目录名}_auto_annotate_labels"
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"

    # 创建输出目录（如果已存在则忽略，同时创建所有必要的父目录）
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # 使用 YOLO 模型进行目标检测
    # stream=True: 逐张图像生成结果，节省内存
    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    # 遍历每张图像的检测结果
    for result in det_results:
        # 使用海象运算符 := 同时赋值和判断
        # 提取类别ID列表，如果列表为空则跳过（没有检测到目标）
        if class_ids := result.boxes.cls.int().tolist():
            # 获取检测框的坐标（xyxy 格式: x1, y1, x2, y2）
            boxes = result.boxes.xyxy

            # 使用 SAM 模型基于检测框生成分割掩码
            # verbose=False: 不打印详细信息
            # save=False: 不保存可视化结果
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)

            # 提取归一化的分割多边形坐标 (xyn: normalized x, y coordinates)
            segments = sam_results[0].masks.xyn

            # 打开输出文件（与输入图像同名的 .txt 文件）
            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w", encoding="utf-8") as f:
                # 遍历每个分割掩码
                for i, s in enumerate(segments):
                    # 检查分割掩码是否有效（不为空）
                    if s.any():
                        # 将分割坐标展平为一维列表并转换为字符串
                        # reshape(-1): 展平为一维数组
                        # tolist(): 转换为 Python 列表
                        # map(str, ...): 将所有数值转换为字符串
                        segment = map(str, s.reshape(-1).tolist())

                        # 写入标注: 格式为 "类别ID x1 y1 x2 y2 ... xn yn\n"
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
