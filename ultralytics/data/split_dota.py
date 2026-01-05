"""
DOTA 数据集拆分模块

DOTA (Dataset for Object deTection in Aerial images) 是一个大规模的航拍图像目标检测数据集。
该模块提供将大尺寸航拍图像切分为小块的功能，同时正确处理旋转边界框标注。

主要功能:
    - 滑动窗口切分大尺寸图像
    - 计算多边形与边界框的 IoF (Intersection over Foreground)
    - 处理 DOTA 数据集的标注格式
    - 支持多尺度切分策略
    - 生成训练、验证和测试集的切片

典型应用场景:
    - DOTA 数据集预处理
    - 航拍图像目标检测数据准备
    - 大图像切块处理
"""

from __future__ import annotations  # 启用延迟类型注解评估

import itertools  # 用于生成笛卡尔积（滑动窗口的所有位置组合）
from glob import glob  # 用于文件路径匹配
from math import ceil  # 向上取整函数
from pathlib import Path  # 跨平台路径操作
from typing import Any  # 类型注解

import cv2  # OpenCV 图像处理库
import numpy as np  # 数值计算库
from PIL import Image  # PIL 图像处理库

# 导入 EXIF 尺寸读取和标签路径转换函数
from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils import TQDM  # 进度条工具
from ultralytics.utils.checks import check_requirements  # 依赖检查工具


def bbox_iof(polygon1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    计算多边形与边界框之间的 IoF (Intersection over Foreground)

    IoF 是交集面积除以前景（多边形）面积，用于判断目标是否被裁剪窗口充分覆盖。

    Calculate Intersection over Foreground (IoF) between polygons and bounding boxes.

    Args:
        polygon1 (np.ndarray): Polygon coordinates with shape (N, 8).
        bbox2 (np.ndarray): Bounding boxes with shape (N, 4).
        eps (float, optional): Small value to prevent division by zero.

    Returns:
        (np.ndarray): IoF scores with shape (N, 1) or (N, M) if bbox2 is (M, 4).

    Notes:
        Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4].
        Bounding box format: [x_min, y_min, x_max, y_max].
    """
    # 检查并确保安装了 shapely 库
    check_requirements("shapely>=2.0.0")
    from shapely.geometry import Polygon

    # 将多边形坐标重塑为 (N, 4, 2) 形状，每个多边形 4 个顶点
    polygon1 = polygon1.reshape(-1, 4, 2)
    # 计算多边形的左上角点（最小 x, y）
    lt_point = np.min(polygon1, axis=-2)
    # 计算多边形的右下角点（最大 x, y）
    rb_point = np.max(polygon1, axis=-2)
    # 将多边形转换为轴对齐的边界框
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    # 计算两个边界框集合之间的重叠区域
    # lt: 交集的左上角点
    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    # rb: 交集的右下角点
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    # wh: 交集的宽高
    wh = np.clip(rb - lt, 0, np.inf)
    # h_overlaps: 交集的面积（近似值，基于边界框）
    h_overlaps = wh[..., 0] * wh[..., 1]

    # 将 bbox2 转换为多边形格式，便于精确计算交集
    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    # 使用 shapely 库创建多边形对象
    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    # 计算精确的交集面积
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    # 计算多边形的面积作为分母
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    # 避免除零错误
    unions = np.clip(unions, eps, np.inf)
    # 计算 IoF = 交集面积 / 前景面积
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def load_yolo_dota(data_root: str, split: str = "train") -> list[dict[str, Any]]:
    """
    加载 DOTA 数据集的标注和图像信息

    Load DOTA dataset annotations and image information.

    Args:
        data_root (str): Data root directory.
        split (str, optional): The split data set, could be 'train' or 'val'.

    Returns:
        (list[dict[str, Any]]): List of annotation dictionaries containing image information.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    # 验证 split 参数
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    # 构建图像目录路径
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    # 获取所有图像文件路径
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    # 根据图像路径生成对应的标签文件路径
    lb_files = img2label_paths(im_files)
    annos = []
    # 遍历每个图像文件及其对应的标签文件
    for im_file, lb_file in zip(im_files, lb_files):
        # 读取图像尺寸（处理 EXIF 旋转信息）
        w, h = exif_size(Image.open(im_file))
        # 读取标签文件
        with open(lb_file, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        # 保存标注信息
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(
    im_size: tuple[int, int],
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
    im_rate_thr: float = 0.6,
    eps: float = 0.01,
) -> np.ndarray:
    """
    获取图像裁剪的滑动窗口坐标

    使用滑动窗口策略生成裁剪位置，支持多尺度裁剪。

    Get the coordinates of sliding windows for image cropping.

    Args:
        im_size (tuple[int, int]): Original image size, (H, W).
        crop_sizes (tuple[int, ...], optional): Crop size of windows.
        gaps (tuple[int, ...], optional): Gap between crops.
        im_rate_thr (float, optional): Threshold of windows areas divided by image areas.
        eps (float, optional): Epsilon value for math operations.

    Returns:
        (np.ndarray): Array of window coordinates of shape (N, 4) where each row is [x_start, y_start, x_stop, y_stop].
    """
    h, w = im_size
    windows = []
    # 遍历每种裁剪尺寸和间隔
    for crop_size, gap in zip(crop_sizes, gaps):
        # 裁剪尺寸必须大于间隔
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        # 滑动步长 = 裁剪尺寸 - 重叠区域
        step = crop_size - gap

        # 计算 x 方向的窗口数量和起始位置
        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        # 调整最后一个窗口，确保不超出图像边界
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        # 计算 y 方向的窗口数量和起始位置
        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        # 调整最后一个窗口，确保不超出图像边界
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        # 生成所有窗口的起始坐标（笛卡尔积）
        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        # 计算窗口的结束坐标
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    # 合并所有尺度的窗口
    windows = np.concatenate(windows, axis=0)

    # 过滤掉超出图像边界过多的窗口
    # 计算窗口内实际图像区域
    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    # 计算实际图像面积
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    # 计算窗口面积
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    # 计算图像面积占窗口面积的比例
    im_rates = im_areas / win_areas
    # 如果没有窗口满足阈值，则保留比例最大的窗口
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    # 返回满足阈值的窗口
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]:
    """
    基于 IoF 阈值获取每个窗口内的目标

    Get objects for each window based on IoF threshold.
    """
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        # 将归一化坐标转换为绝对坐标
        label[:, 1::2] *= w  # x 坐标
        label[:, 2::2] *= h  # y 坐标
        # 计算每个目标与所有窗口的 IoF
        iofs = bbox_iof(label[:, 1:], windows)
        # 返回每个窗口内 IoF 大于阈值的目标（未归一化和未对齐的坐标）
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]
    else:
        # 如果没有标签，返回空数组列表
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]


def crop_and_save(
    anno: dict[str, Any],
    windows: np.ndarray,
    window_objs: list[np.ndarray],
    im_dir: str,
    lb_dir: str,
    allow_background_images: bool = True,
) -> None:
    """
    裁剪图像并为每个窗口保存新标签

    Crop images and save new labels for each window.

    Args:
        anno (dict[str, Any]): Annotation dict, including 'filepath', 'label', 'ori_size' as its keys.
        windows (np.ndarray): Array of windows coordinates with shape (N, 4).
        window_objs (list[np.ndarray]): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.
        allow_background_images (bool, optional): Whether to include background images without labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    # 读取原始图像
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    # 遍历每个窗口
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        # 生成新文件名，格式: 原名__窗口大小__x坐标___y坐标
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        # 裁剪图像块
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        # 获取当前窗口内的标签
        label = window_objs[i]
        # 如果有标签或允许保存背景图像，则保存图像
        if len(label) or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        # 如果有标签，转换坐标并保存
        if len(label):
            # 将绝对坐标转换为相对于窗口的坐标
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            # 归一化坐标到 [0, 1]
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

            # 保存标签文件
            with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                for lb in label:
                    # 格式化坐标，保留 6 位有效数字
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def split_images_and_labels(
    data_root: str,
    save_dir: str,
    split: str = "train",
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
) -> None:
    """
    拆分指定数据集划分的图像和标签

    Split both images and labels for a given dataset split.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        split (str, optional): The split data set, could be 'train' or 'val'.
        crop_sizes (tuple[int, ...], optional): Tuple of crop sizes.
        gaps (tuple[int, ...], optional): Tuple of gaps between crops.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    # 创建输出目录
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注数据
    annos = load_yolo_dota(data_root, split=split)
    # 处理每张图像
    for anno in TQDM(annos, total=len(annos), desc=split):
        # 获取滑动窗口
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        # 获取每个窗口内的目标
        window_objs = get_window_obj(anno, windows)
        # 裁剪并保存
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """
    使用多尺度策略拆分 DOTA 数据集的训练集和验证集

    Split train and val sets of DOTA dataset with multiple scaling rates.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        crop_size (int, optional): Base crop size.
        gap (int, optional): Base gap between crops.
        rates (tuple[float, ...], optional): Scaling rates for crop_size and gap.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    # 根据缩放率计算不同尺度的裁剪尺寸和间隔
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    # 分别处理训练集和验证集
    for split in {"train", "val"}:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


def split_test(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """
    拆分 DOTA 数据集的测试集（测试集不包含标签）

    Split test set of DOTA dataset, labels are not included within this set.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        crop_size (int, optional): Base crop size.
        gap (int, optional): Base gap between crops.
        rates (tuple[float, ...], optional): Scaling rates for crop_size and gap.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test
    """
    # 根据缩放率计算不同尺度的裁剪尺寸和间隔
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    # 创建输出目录
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试集图像路径
    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    # 处理每张测试图像
    for im_file in TQDM(im_files, total=len(im_files), desc="test"):
        # 读取图像尺寸
        w, h = exif_size(Image.open(im_file))
        # 生成滑动窗口
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        # 读取图像
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        # 裁剪并保存每个窗口
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")
