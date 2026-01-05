"""
数据集拆分模块

该模块提供数据集拆分功能，支持将数据集按比例划分为训练集、验证集和测试集。
主要用于分类任务和目标检测任务的数据准备。

主要功能:
    - split_classify_dataset: 拆分分类数据集为训练集和验证集
    - autosplit: 自动拆分目标检测数据集，生成 autosplit_*.txt 文件

典型应用场景:
    - 准备训练数据集
    - 数据集预处理
    - 自动化数据管理
"""

from __future__ import annotations  # 启用延迟类型注解评估

import random  # 用于随机打乱数据
import shutil  # 用于文件复制操作
from pathlib import Path  # 跨平台路径操作

# 导入图像格式常量和标签路径转换函数
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
# 导入数据集目录常量、日志记录器和进度条工具
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """
    拆分分类数据集为训练集和验证集

    在新目录中创建 train/val 子目录，保持原有的类别目录结构。
    默认按 80/20 比例拆分。

    Split classification dataset into train and val directories in a new directory.

    Creates a new directory '{source_dir}_split' with train/val subdirectories, preserving the original class structure
    with an 80/20 split by default.

    Directory structure:
        Before:
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        After:
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    Args:
        source_dir (str | Path): Path to classification dataset root directory.
        train_ratio (float): Ratio for train split, between 0 and 1.

    Returns:
        (Path): Path to the created split directory.

    Examples:
        Split dataset with default 80/20 ratio
        >>> split_classify_dataset("path/to/caltech")

        Split with custom ratio
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    # 转换为 Path 对象
    source_path = Path(source_dir)
    # 创建输出目录路径，在原目录名后添加 "_split" 后缀
    split_path = Path(f"{source_path}_split")
    # 定义训练集和验证集目录路径
    train_path, val_path = split_path / "train", split_path / "val"

    # 创建目录结构
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # 处理类别目录
    # 获取所有类别目录（过滤掉文件，只保留目录）
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    # 统计总图像数
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    # 遍历每个类别目录
    for class_dir in class_dirs:
        # 在训练集和验证集目录中创建对应的类别目录
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # 拆分并复制文件
        # 获取当前类别的所有图像文件
        image_files = list(class_dir.glob("*.*"))
        # 随机打乱文件顺序
        random.shuffle(image_files)
        # 计算训练集和验证集的分割点
        split_idx = int(len(image_files) * train_ratio)

        # 复制训练集图像（前 split_idx 个）
        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        # 复制验证集图像（剩余的）
        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """
    自动拆分数据集为训练集/验证集/测试集并保存到 autosplit_*.txt 文件

    该函数会在图像目录的父目录中生成三个文本文件，每个文件包含对应拆分的图像路径列表。
    默认按 90/10/0 比例拆分（90% 训练集，10% 验证集，0% 测试集）。

    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt
    files.

    Args:
        path (Path): Path to images directory.
        weights (tuple): Train, validation, and test split fractions.
        annotated_only (bool): If True, only images with an associated txt file are used.

    Examples:
        Split images with default weights
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        Split with custom weights and annotated images only
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    # 转换为 Path 对象
    path = Path(path)
    # 递归查找所有支持的图像文件，并按名称排序
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    n = len(files)
    # 设置随机种子以保证结果可复现
    random.seed(0)
    # 为每张图像随机分配到训练集(0)、验证集(1)或测试集(2)
    indices = random.choices([0, 1, 2], weights=weights, k=n)

    # 定义三个输出文本文件的名称
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    # 删除已存在的拆分文件
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    # 遍历所有图像，根据分配的索引写入对应的文件
    for i, img in TQDM(zip(indices, files), total=n):
        # 如果 annotated_only 为 True，则检查标签文件是否存在
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            # 将图像的相对路径追加到对应的文本文件中
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")


if __name__ == "__main__":
    split_classify_dataset("caltech101")
