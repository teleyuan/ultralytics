# Ultralytics YOLO 数据集基础类模块
# 该模块提供了数据集加载、缓存和处理的核心功能

from __future__ import annotations  # 启用未来注解特性，允许使用字符串形式的类型提示

import glob  # 文件模式匹配工具
import math  # 数学函数库
import os  # 操作系统接口
import random  # 随机数生成
from copy import deepcopy  # 深拷贝函数
from multiprocessing.pool import ThreadPool  # 多线程池
from pathlib import Path  # 面向对象的文件路径操作
from typing import Any  # 类型提示

import cv2  # OpenCV 计算机视觉库
import numpy as np  # 数值计算库
from torch.utils.data import Dataset  # PyTorch 数据集基类

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds  # 数据工具函数
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM  # 通用工具
from ultralytics.utils.patches import imread  # 图像读取函数


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.
    用于加载和处理图像数据的基础数据集类。

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.
    该类提供了加载图像、缓存和为目标检测任务的训练和推理准备数据的核心功能。

    Attributes:
        img_path (str): Path to the folder containing images.
            图像文件夹路径。
        imgsz (int): Target image size for resizing.
            调整大小后的目标图像尺寸。
        augment (bool): Whether to apply data augmentation.
            是否应用数据增强。
        single_cls (bool): Whether to treat all objects as a single class.
            是否将所有对象视为单一类别。
        prefix (str): Prefix to print in log messages.
            日志消息中打印的前缀。
        fraction (float): Fraction of dataset to utilize.
            使用数据集的比例。
        channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV
            are in BGR channel order.
            图像的通道数（1表示灰度图，3表示彩色图）。使用OpenCV加载的彩色图像采用BGR通道顺序。
        cv2_flag (int): OpenCV flag for reading images.
            OpenCV读取图像的标志。
        im_files (list[str]): List of image file paths.
            图像文件路径列表。
        labels (list[dict]): List of label data dictionaries.
            标签数据字典列表。
        ni (int): Number of images in the dataset.
            数据集中的图像数量。
        rect (bool): Whether to use rectangular training.
            是否使用矩形训练。
        batch_size (int): Size of batches.
            批次大小。
        stride (int): Stride used in the model.
            模型中使用的步长。
        pad (float): Padding value.
            填充值。
        buffer (list): Buffer for mosaic images.
            马赛克图像的缓冲区。
        max_buffer_length (int): Maximum buffer size.
            最大缓冲区大小。
        ims (list): List of loaded images.
            已加载图像的列表。
        im_hw0 (list): List of original image dimensions (h, w).
            原始图像尺寸列表（高度，宽度）。
        im_hw (list): List of resized image dimensions (h, w).
            调整大小后图像尺寸列表（高度，宽度）。
        npy_files (list[Path]): List of numpy file paths.
            numpy文件路径列表。
        cache (str): Cache images to RAM or disk during training.
            训练期间将图像缓存到RAM或磁盘。
        transforms (callable): Image transformation function.
            图像转换函数。
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
            矩形训练的批次形状。
        batch (np.ndarray): Batch index of each image.
            每个图像的批次索引。

    Methods:
        get_img_files: Read image files from the specified path.
            从指定路径读取图像文件。
        update_labels: Update labels to include only specified classes.
            更新标签以仅包含指定的类别。
        load_image: Load an image from the dataset.
            从数据集加载图像。
        cache_images: Cache images to memory or disk.
            将图像缓存到内存或磁盘。
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
            将图像保存为*.npy文件以加快加载速度。
        check_cache_disk: Check image caching requirements vs available disk space.
            检查图像缓存需求与可用磁盘空间。
        check_cache_ram: Check image caching requirements vs available memory.
            检查图像缓存需求与可用内存。
        set_rectangle: Set the shape of bounding boxes as rectangles.
            设置边界框的形状为矩形。
        get_image_and_label: Get and return label information from the dataset.
            从数据集获取并返回标签信息。
        update_labels_info: Custom label format method to be implemented by subclasses.
            由子类实现的自定义标签格式方法。
        build_transforms: Build transformation pipeline to be implemented by subclasses.
            由子类实现的构建转换流水线。
        get_labels: Get labels method to be implemented by subclasses.
            由子类实现的获取标签方法。
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """
        Initialize BaseDataset with given configuration and options.
        使用给定的配置和选项初始化BaseDataset。

        Args:
            img_path (str | list[str]): Path to the folder containing images or list of image paths.
                包含图像的文件夹路径或图像路径列表。
            imgsz (int): Image size for resizing.
                调整大小的图像尺寸。
            cache (bool | str): Cache images to RAM or disk during training.
                训练期间将图像缓存到RAM或磁盘。
            augment (bool): If True, data augmentation is applied.
                如果为True，则应用数据增强。
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation.
                应用数据增强的超参数。
            prefix (str): Prefix to print in log messages.
                日志消息中打印的前缀。
            rect (bool): If True, rectangular training is used.
                如果为True，则使用矩形训练。
            batch_size (int): Size of batches.
                批次大小。
            stride (int): Stride used in the model.
                模型中使用的步长。
            pad (float): Padding value.
                填充值。
            single_cls (bool): If True, single class training is used.
                如果为True，则使用单类别训练。
            classes (list[int], optional): List of included classes.
                包含的类别列表（可选）。
            fraction (float): Fraction of dataset to utilize.
                使用数据集的比例。
            channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with
                OpenCV are in BGR channel order.
                图像的通道数（1表示灰度图，3表示彩色图）。使用OpenCV加载的彩色图像采用BGR通道顺序。
        """
        super().__init__()
        self.img_path = img_path  # 图像路径
        self.imgsz = imgsz  # 目标图像尺寸
        self.augment = augment  # 是否数据增强
        self.single_cls = single_cls  # 是否单类别模式
        self.prefix = prefix  # 日志前缀
        self.fraction = fraction  # 数据集使用比例
        self.channels = channels  # 图像通道数
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # 设置OpenCV读取标志
        self.im_files = self.get_img_files(self.img_path)  # 获取图像文件列表
        self.labels = self.get_labels()  # 获取标签
        self.update_labels(include_class=classes)  # 更新标签，处理单类别和类别过滤
        self.ni = len(self.labels)  # 数据集图像数量
        self.rect = rect  # 是否矩形训练
        self.batch_size = batch_size  # 批次大小
        self.stride = stride  # 模型步长
        self.pad = pad  # 填充值
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()  # 设置矩形训练参数

        # 马赛克图像的缓冲线程
        self.buffer = []  # 缓冲区列表，用于存储图像索引
        # 计算最大缓冲区长度：如果使用增强，取(图像数量, 批次大小*8, 1000)中的最小值，否则为0
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # 缓存图像（选项有：cache = True, False, None, "ram", "disk"）
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni  # 初始化图像缓存列表
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]  # numpy缓存文件路径列表
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None  # 规范化缓存选项
        if self.cache == "ram" and self.check_cache_ram():  # RAM缓存
            if hyp.deterministic:  # 如果需要确定性训练
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()  # 缓存图像到RAM
        elif self.cache == "disk" and self.check_cache_disk():  # 磁盘缓存
            self.cache_images()  # 缓存图像到磁盘

        # 图像转换
        self.transforms = self.build_transforms(hyp=hyp)  # 构建转换流水线

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """
        Read image files from the specified path.
        从指定路径读取图像文件。

        Args:
            img_path (str | list[str]): Path or list of paths to image directories or files.
                图像目录或文件的路径或路径列表。

        Returns:
            (list[str]): List of image file paths.
                图像文件路径列表。

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
                如果未找到图像或路径不存在，则引发此异常。
        """
        try:
            f = []  # 图像文件列表
            for p in img_path if isinstance(img_path, list) else [img_path]:  # 遍历路径列表
                p = Path(p)  # 转换为Path对象，跨平台兼容
                if p.is_dir():  # 如果是目录
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)  # 递归搜索所有文件
                    # F = list(p.rglob('*.*'))  # pathlib方式
                elif p.is_file():  # 如果是文件（包含图像路径的文本文件）
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()  # 读取文件内容并按行分割
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # 将相对路径转换为全局路径
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # pathlib方式
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            # 过滤出有效的图像文件并排序
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib方式
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:  # 如果只使用部分数据集
            im_files = im_files[: round(len(im_files) * self.fraction)]  # 保留数据集的指定比例
        check_file_speeds(im_files, prefix=self.prefix)  # 检查图像读取速度
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """
        Update labels to include only specified classes.
        更新标签以仅包含指定的类别。

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
                要包含的类别列表（可选）。如果为None，则包含所有类别。
        """
        include_class_array = np.array(include_class).reshape(1, -1)  # 将类别列表转换为numpy数组
        for i in range(len(self.labels)):  # 遍历所有标签
            if include_class is not None:  # 如果指定了要包含的类别
                cls = self.labels[i]["cls"]  # 获取类别
                bboxes = self.labels[i]["bboxes"]  # 获取边界框
                segments = self.labels[i]["segments"]  # 获取分割掩码
                keypoints = self.labels[i]["keypoints"]  # 获取关键点
                j = (cls == include_class_array).any(1)  # 找出属于指定类别的对象
                self.labels[i]["cls"] = cls[j]  # 过滤类别
                self.labels[i]["bboxes"] = bboxes[j]  # 过滤边界框
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]  # 过滤分割掩码
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]  # 过滤关键点
            if self.single_cls:  # 如果是单类别模式
                self.labels[i]["cls"][:, 0] = 0  # 将所有类别设置为0

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index 'i'.
        从数据集索引'i'加载图像。

        Args:
            i (int): Index of the image to load.
                要加载的图像索引。
            rect_mode (bool): Whether to use rectangular resizing.
                是否使用矩形调整大小。

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
                加载的图像（NumPy数组）。
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
                原始图像尺寸（高度，宽度）格式。
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.
                调整大小后的图像尺寸（高度，宽度）格式。

        Raises:
            FileNotFoundError: If the image file is not found.
                如果未找到图像文件，则引发此异常。
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]  # 获取缓存图像、文件路径和npy文件路径
        if im is None:  # 如果图像未缓存到RAM
            if fn.exists():  # 如果存在npy缓存文件
                try:
                    im = np.load(fn)  # 从npy文件加载图像
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)  # 删除损坏的npy文件
                    im = imread(f, flags=self.cv2_flag)  # 从原始图像文件读取（BGR格式）
            else:  # 读取图像
                im = imread(f, flags=self.cv2_flag)  # BGR格式
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # 原始高度和宽度
            if rect_mode:  # 保持宽高比的情况下，将长边调整为imgsz
                r = self.imgsz / max(h0, w0)  # 缩放比例
                if r != 1:  # 如果尺寸不相等
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)  # 线性插值调整大小
            elif not (h0 == w0 == self.imgsz):  # 通过拉伸将图像调整为正方形imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:  # 如果是灰度图
                im = im[..., None]  # 添加通道维度

            # 如果使用增强训练，则添加到缓冲区
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # 存储图像、原始尺寸、调整后尺寸
                self.buffer.append(i)  # 添加到缓冲区
                if 1 < len(self.buffer) >= self.max_buffer_length:  # 防止缓冲区为空
                    j = self.buffer.pop(0)  # 移除最早的图像
                    if self.cache != "ram":  # 如果不是RAM缓存
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None  # 清除缓存

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # 返回缓存的图像

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
