"""
Ultralytics 工具模块初始化文件

这是 Ultralytics YOLO 库的核心工具模块，提供了大量通用的辅助函数、类和常量。
该模块包含了项目中广泛使用的工具函数，涵盖文件操作、系统检测、日志管理、配置处理等多个方面。

主要功能:
    - 系统环境检测 (操作系统、设备类型、运行环境等)
    - 日志和输出管理 (彩色输出、日志记录器配置等)
    - 配置文件处理 (YAML 加载保存、设置管理等)
    - 多线程和异常处理 (线程锁、重试机制、异常捕获等)
    - 平台相关常量 (路径、版本、设备信息等)
    - 数据导出工具 (CSV、JSON、DataFrame 等)

常量定义:
    - RANK/LOCAL_RANK: 分布式训练进程标识
    - ROOT/ASSETS: 项目根目录和资源路径
    - MACOS/LINUX/WINDOWS: 操作系统标识
    - IS_COLAB/IS_KAGGLE/IS_DOCKER: 运行环境标识
    - SETTINGS: 全局设置管理器

核心类:
    - YAML: YAML 文件操作工具类
    - SettingsManager: 设置管理类
    - SimpleClass: 简单基类，提供友好的字符串表示
    - IterableSimpleNamespace: 可迭代的命名空间类
    - ThreadingLocked: 线程安全装饰器
    - TryExcept/Retry: 异常处理和重试装饰器
"""

from __future__ import annotations  # 支持 Python 3.7+ 的类型注解语法

# 标准库导入
import contextlib  # 上下文管理器工具
import importlib.metadata  # 包元数据访问
import inspect  # 代码检查和内省
import json  # JSON 编码解码
import logging  # 日志记录
import os  # 操作系统接口
import platform  # 平台信息
import re  # 正则表达式
import socket  # 网络接口
import sys  # 系统相关参数和函数
import threading  # 线程操作
import time  # 时间相关函数
import warnings  # 警告控制
from functools import lru_cache  # LRU 缓存装饰器
from pathlib import Path  # 面向对象的文件系统路径
from threading import Lock  # 线程锁
from types import SimpleNamespace  # 简单的命名空间对象
from urllib.parse import unquote  # URL 解码

# 第三方库导入
import cv2  # OpenCV 计算机视觉库
import numpy as np  # NumPy 数值计算库
import torch  # PyTorch 深度学习框架

# Ultralytics 内部导入
from ultralytics import __version__  # YOLO 版本号
from ultralytics.utils.git import GitRepo  # Git 仓库操作类
from ultralytics.utils.patches import imread, imshow, imwrite, torch_save  # 修补函数 (解决路径编码问题)
from ultralytics.utils.tqdm import TQDM  # 进度条工具 (增强版)  # noqa

# ===================================================================================================
# 常量定义区域
# ===================================================================================================

# PyTorch 多 GPU 分布式数据并行 (DDP) 常量
RANK = int(os.getenv("RANK", -1))  # 当前进程在所有进程中的全局排名，-1 表示非分布式训练
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 当前进程在本机所有进程中的排名 (单机多卡时使用)
# 参考: https://pytorch.org/docs/stable/elastic/run.html

# 路径和文件相关常量
ARGV = sys.argv or ["", ""]  # 命令行参数列表 (有时 sys.argv 可能为空列表，需要默认值)
FILE = Path(__file__).resolve()  # 当前文件的绝对路径
ROOT = FILE.parents[1]  # Ultralytics 项目根目录 (向上两级目录)
ASSETS = ROOT / "assets"  # 默认图片和资源文件目录
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # 资源文件的 GitHub 下载地址
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"  # 默认配置文件路径

# 性能和行为相关常量
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # YOLO 多进程处理的线程数 (最少1个，最多8个)
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == "true"  # 是否自动安装缺失的依赖包
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"  # 是否启用详细输出模式
LOGGING_NAME = "ultralytics"  # 日志记录器名称

# 操作系统和平台检测
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # 操作系统布尔标志
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None  # macOS 版本号 (仅在 macOS 上有效)
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))  # 是否不是 macOS 14 (用于兼容性检查)
ARM64 = platform.machine() in {"arm64", "aarch64"}  # 是否为 ARM64 架构 (如 Apple Silicon, Raspberry Pi)

# 软件版本信息
PYTHON_VERSION = platform.python_version()  # Python 版本字符串
TORCH_VERSION = str(torch.__version__)  # PyTorch 版本 (标准化为字符串，因为 PyTorch>1.9 返回 TorchVersion 对象)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # torchvision 版本 (直接读取元数据，比导入模块更快)

# 开发环境检测
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"  # 是否在 VS Code 终端中运行

# 硬件芯片支持 - Rockchip NPU 芯片列表
RKNN_CHIPS = frozenset(
    {
        "rk3588",  # 瑞芯微 RK3588 (高性能 SoC)
        "rk3576",  # 瑞芯微 RK3576
        "rk3566",  # 瑞芯微 RK3566
        "rk3568",  # 瑞芯微 RK3568
        "rk3562",  # 瑞芯微 RK3562
        "rv1103",  # 瑞芯微 RV1103
        "rv1106",  # 瑞芯微 RV1106
        "rv1103b",  # 瑞芯微 RV1103B
        "rv1106b",  # 瑞芯微 RV1106B
        "rk2118",  # 瑞芯微 RK2118
        "rv1126b",  # 瑞芯微 RV1126B
    }
)  # 支持导出 RKNN 格式的 Rockchip 处理器芯片型号集合 (frozenset 为不可变集合，提升性能)
HELP_MSG = """
    Examples for running Ultralytics:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.yaml")  # build a new model from scratch
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco8.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        success = model.export(format="onnx")  # export the model to ONNX format

    3. Use the command line interface (CLI):

        Ultralytics 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                    MODE (required) is one of [train, val, predict, export, track, benchmark]
                    ARGS (optional) are any number of custom "arg=value" pairs like "imgsz=320" that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with "yolo cfg"

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

        - Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# ===================================================================================================
# 全局设置和环境变量配置
# ===================================================================================================

# PyTorch 打印选项配置
torch.set_printoptions(linewidth=320, precision=4, profile="default")
# linewidth=320: 每行最多显示 320 个字符
# precision=4: 浮点数精度为 4 位小数
# profile="default": 使用默认打印配置

# NumPy 打印选项配置
np.set_printoptions(linewidth=320, formatter=dict(float_kind="{:11.5g}".format))
# linewidth=320: 每行最多显示 320 个字符
# formatter: 浮点数格式化为科学计数法，总宽度 11，有效数字 5 位

# OpenCV 线程设置
cv2.setNumThreads(0)
# 禁用 OpenCV 的多线程 (与 PyTorch DataLoader 的多进程不兼容，可能导致死锁)

# 环境变量设置 - 控制第三方库的行为
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr 库的最大线程数
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 抑制 TensorFlow 在 Colab 中的详细编译警告 (3=仅显示错误)
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # 抑制 PyTorch 的 "NNPACK.cpp could not initialize NNPACK" 警告
os.environ["KINETO_LOG_LEVEL"] = "5"  # 抑制计算 FLOPs 时 PyTorch profiler 的详细输出

# ===================================================================================================
# 集中式警告过滤 - 抑制已知的无害警告
# ===================================================================================================

# PyTorch 相关警告
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")  # PyTorch 弃用警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # ONNX/TorchScript 导出时的追踪警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")  # ONNX 形状推断警告

# 第三方库警告
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")  # matplotlib>=3.7.2 布局变更警告
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")  # timm 库 (mobileclip) 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")  # CoreML 的 np.bool 弃用警告

# ===================================================================================================
# 预编译的类型元组 - 用于加速 isinstance() 类型检查
# ===================================================================================================

FLOAT_OR_INT = (float, int)  # 数值类型 (浮点数或整数)
STR_OR_PATH = (str, Path)  # 字符串或路径类型


class DataExportMixin:
    """
    数据导出混入类 - 用于将验证指标或预测结果导出为多种格式

    Mixin class for exporting validation metrics or prediction results in various formats.

    该类提供了将性能指标 (如 mAP、精确率、召回率) 或预测结果 (分类、目标检测、分割、姿态估计等任务)
    导出为多种格式的工具方法: Polars DataFrame、CSV 和 JSON。

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Polars
    DataFrame, CSV, and JSON.

    方法:
        to_df: 将摘要转换为 Polars DataFrame
        to_csv: 将结果导出为 CSV 字符串
        to_json: 将结果导出为 JSON 字符串

    Methods:
        to_df: Convert summary to a Polars DataFrame.
        to_csv: Export results as a CSV string.
        to_json: Export results as a JSON string.

    示例:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()

    Examples:
        >>> model = YOLO("yolo11n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """
        从预测结果摘要或验证指标创建 Polars DataFrame
        Create a Polars DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): 是否归一化数值以便于比较。默认 False
            decimals (int, optional): 浮点数保留的小数位数。默认 5

        Returns:
            (polars.DataFrame): 包含摘要数据的 Polars DataFrame
        """
        import polars as pl  # 局部导入以加快 'import ultralytics' 速度

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """
        将结果或指标导出为 CSV 字符串格式
        Export results or metrics to CSV string format.

        Args:
            normalize (bool, optional): 是否归一化数值。默认 False
            decimals (int, optional): 小数精度。默认 5

        Returns:
            (str): CSV 内容字符串
        """
        import polars as pl  # 局部导入

        df = self.to_df(normalize=normalize, decimals=decimals)

        try:
            # 尝试直接写入 CSV
            return df.write_csv()
        except Exception:
            # 如果有复杂类型无法序列化，进行最小化字符串转换
            def _to_str_simple(v):
                """简单的字符串转换函数"""
                if v is None:
                    return ""  # None 转为空字符串
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)  # 复杂类型转为字符串表示
                else:
                    return str(v)  # 其他类型直接转字符串

            # 将所有列转换为字符串类型
            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """
        将结果导出为 JSON 格式
        Export results to JSON format.

        Args:
            normalize (bool, optional): 是否归一化数值。默认 False
            decimals (int, optional): 小数精度。默认 5

        Returns:
            (str): 结果的 JSON 格式字符串
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:
    """
    简单基类 - 为对象提供友好的字符串表示
    A simple base class for creating objects with string representations of their attributes.

    该类为创建可轻松打印或表示为字符串的对象提供基础，显示所有非可调用属性。
    对于调试和检查对象状态非常有用。

    This class provides a foundation for creating objects that can be easily printed or represented as strings, showing
    all their non-callable attributes. It's useful for debugging and introspection of object states.

    方法:
        __str__: 返回对象的人类可读字符串表示
        __repr__: 返回对象的机器可读字符串表示
        __getattr__: 提供自定义的属性访问错误消息

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    示例:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    注意:
        - 该类设计用于被子类化，提供方便的对象属性检查方式
        - 字符串表示包含对象的模块名和类名
        - 可调用属性和以下划线开头的属性不会显示在字符串表示中

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """
        返回对象的人类可读字符串表示
        Return a human-readable string representation of the object.
        """
        attr = []
        for a in dir(self):  # 遍历对象的所有属性
            v = getattr(self, a)  # 获取属性值
            if not callable(v) and not a.startswith("_"):  # 排除方法和私有属性
                if isinstance(v, SimpleClass):
                    # 如果属性值也是 SimpleClass 的子类，只显示模块名和类名
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    # 其他类型使用 repr() 显示
                    s = f"{a}: {v!r}"
                attr.append(s)
        # 返回格式化的属性列表
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """
        返回对象的机器可读字符串表示
        Return a machine-readable string representation of the object.
        """
        return self.__str__()

    def __getattr__(self, attr):
        """
        提供自定义的属性访问错误消息
        Provide a custom attribute access error message with helpful information.
        """
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation, and
    attribute access. It is designed to be used as a convenient container for storing and accessing configuration
    parameters.

    Methods:
        __iter__: Return an iterator of key-value pairs from the namespace's attributes.
        __str__: Return a human-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.
        get: Retrieve the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend="Agg"):
    """Decorator to temporarily set rc parameters and the backend for a plotting function.

    Args:
        rcparams (dict, optional): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend.

    Examples:
        >>> @plt_settings({"font.size": 12})
        >>> def plot_function():
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()

        >>> with plt_settings({"font.size": 12}):
        ...     plt.figure()
        ...     plt.plot([1, 2, 3])
        ...     plt.show()
    """
    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Set rc parameters and backend, call the original function, and restore the settings."""
            import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            # Plot with backend and always revert to original backend
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    """Set up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and formatter
    based on the verbosity flag and the current process rank. It handles special cases for Windows environments where
    UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """Format log records with prefixes based on level."""
            # Apply prefixes based on log level
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"

            # Handle emojis in message based on platform
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # Handle Windows UTF-8 encoding issues
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


class ThreadingLocked:
    """A decorator class for ensuring thread-safe execution of a function or method.

    This class can be used as a decorator to make sure that if the decorated function is called from multiple threads,
    only one thread at a time will be able to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Examples:
        >>> from ultralytics.utils import ThreadingLocked
        >>> @ThreadingLocked()
        >>> def my_function():
        ...    # Your code here
    """

    def __init__(self):
        """Initialize the decorator class with a threading lock."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Apply thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated


class YAML:
    """YAML utility class for efficient file operations with automatic C-implementation detection.

    This class provides optimized YAML loading and saving operations using PyYAML's fastest available implementation
    (C-based when possible). It implements a singleton pattern with lazy initialization, allowing direct class method
    usage without explicit instantiation. The class handles file path creation, validation, and character encoding
    issues automatically.

    The implementation prioritizes performance through:
        - Automatic C-based loader/dumper selection when available
        - Singleton pattern to reuse the same instance
        - Lazy initialization to defer import costs until needed
        - Fallback mechanisms for handling problematic YAML content

    Attributes:
        _instance: Internal singleton instance storage.
        yaml: Reference to the PyYAML module.
        SafeLoader: Best available YAML loader (CSafeLoader if available).
        SafeDumper: Best available YAML dumper (CSafeDumper if available).

    Examples:
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """Initialize singleton instance on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize with optimal YAML implementation (C-based when available)."""
        import yaml

        self.yaml = yaml
        # Use C-based implementation if available for better performance
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file="data.yaml", data=None, header=""):
        """Save Python object as YAML file.

        Args:
            file (str | Path): Path to save YAML file.
            data (dict | None): Dict or compatible object to save.
            header (str): Optional string to add at file beginning.
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # Create parent directories if needed
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects to strings
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # Write YAML file
        with open(file, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file="data.yaml", append_filename=False):
        """Load YAML file to Python object with robust error handling.

        Args:
            file (str | Path): Path to YAML file.
            append_filename (bool): Whether to add filename to returned dict.

        Returns:
            (dict): Loaded YAML content.
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"Not a YAML file: {file}"

        # Read file content
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # Try loading YAML with fallback for problematic characters
        try:
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception:
            # Remove problematic characters and retry
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

        # Check for accidental user-error None strings (should be 'null' in YAML)
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file):
        """Pretty print YAML file or object to console.

        Args:
            yaml_file (str | Path | dict): Path to YAML file or dict to print.
        """
        instance = cls._get_instance()

        # Load file if path provided
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

        # Use -1 for unlimited width in C implementation
        dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

        LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def read_device_model() -> str:
    """Read the device model information from the system and cache it for quick access.

    Returns:
        (str): Kernel release information.
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: list[str] | None | str = None) -> list[bool] | bool:
    """Check if the OS is Debian.

    Args:
        codenames (list[str] | None | str): Specific Debian codename to check for (e.g., 'buster', 'bullseye'). If None,
            only checks for Debian.

    Returns:
        (list[bool] | bool): List of booleans indicating if OS matches each Debian codename, or a single boolean if no
            codenames provided.
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    """Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """Check if the current script is running inside a Jupyter Notebook.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.

    Notes:
        - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
        - "get_ipython" in globals() method suffers false positives when IPython package installed manually.
    """
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    """Check if the current script is running inside a RunPod container.

    Returns:
        (bool): True if running in RunPod, False otherwise.
    """
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    """Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def is_raspberrypi() -> bool:
    """Determine if the Python environment is running on a Raspberry Pi.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    return "rpi" in DEVICE_MODEL


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """Determine if the Python environment is running on an NVIDIA Jetson device.

    Args:
        jetpack (int | None): If specified, check for specific JetPack version (4, 5, 6).

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = open("/etc/nv_tegra_release").read()
            version_map = {4: "R32", 5: "R35", 6: "R36"}  # JetPack to L4T major version mapping
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


def is_online() -> bool:
    """Fast online check using DNS (v4/v6) resolution (Cloudflare + Google).

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    if str(os.getenv("YOLO_OFFLINE", "")).lower() == "true":
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False


def is_pip_package(filepath: str = __name__) -> bool:
    """Determine if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: str | Path) -> bool:
    """Check if a directory is writable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """Determine whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)


def is_github_action_running() -> bool:
    """Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_default_args(func):
    """Return a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    """Return a writable config dir, preferring YOLO_CONFIG_DIR and being OS-aware.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    if p.exists():  # already created → trust it
        return p
    if is_dir_writeable(p.parent):  # create if possible
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Fallbacks for Docker, GCP/AWS functions where only /tmp is writable
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt

    # Last fallback → CWD
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


# Define constants (required below)
DEVICE_MODEL = read_device_model()  # is_jetson() and is_raspberrypi() depend on this constant
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_JUPYTER = is_jupyter()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
IS_DEBIAN, IS_DEBIAN_BOOKWORM, IS_DEBIAN_TRIXIE = is_debian([None, "bookworm", "trixie"])
IS_UBUNTU = is_ubuntu()
GIT = GitRepo()
USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


def colorstr(*input):
    r"""Color a string based on the provided color and style arguments using ANSI escape codes.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments, and the
            last string is the one to be colored.

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"

    Notes:
        Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def remove_colorstr(input_string):
    """Remove ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        >>> "hello world"
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Execute when entering TryExcept context, initialize instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Define behavior when exiting a 'with' block, print error message if necessary."""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """Retry class for function execution with exponential backoff.

    This decorator can be used to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries. It's useful for handling transient failures in network operations or
    other unreliable processes.

    Attributes:
        times (int): Maximum number of retry attempts.
        delay (int): Initial delay between retries in seconds.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>> # Replace with function logic that may raise exceptions
        >>>     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Apply retries to the decorated function or method."""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

        return wrapped_func


def threaded(func):
    """Multi-thread a target function by default and return the thread or function result.

    This decorator provides flexible execution of the target function, either in a separate thread or synchronously. By
    default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument which
    is removed from kwargs before calling the function.

    Args:
        func (callable): The function to be potentially executed in a separate thread.

    Returns:
        (callable): A wrapper function that either returns a daemon thread or the direct function result.

    Examples:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # Runs in background thread
        >>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
    """

    def wrapper(*args, **kwargs):
        """Multi-thread a given function based on 'threaded' kwarg and return the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def set_sentry():
    """Initialize the Sentry SDK for error tracking and reporting.

    Only used if sentry_sdk package is installed and sync=True in settings. Run 'yolo settings' to see and update
    settings.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)
    """
    if (
        not SETTINGS["sync"]
        or RANK not in {-1, 0}
        or Path(ARGV[0]).name != "yolo"
        or TESTS_RUNNING
        or not ONLINE
        or not IS_PIP_PACKAGE
        or GIT.is_repo
    ):
        return
    # If sentry_sdk package is not installed then return and do not use Sentry
    try:
        import sentry_sdk
    except ImportError:
        return

    def before_send(event, hint):
        """Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            (dict | None): The modified event or None if the event should not be sent to Sentry.
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if exc_type in {KeyboardInterrupt, FileNotFoundError} or "out of memory" in str(exc_value):
                return None  # do not send event

        event["tags"] = {
            "sys_argv": ARGV[0],
            "sys_argv_name": Path(ARGV[0]).name,
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "os": ENVIRONMENT,
        }
        return event

    sentry_sdk.init(
        dsn="https://888e5a0778212e1d0314c37d4b9aae5d@o4504521589325824.ingest.us.sentry.io/4504521592406016",
        debug=False,
        auto_enabling_integrations=False,
        traces_sample_rate=1.0,
        release=__version__,
        environment="runpod" if is_runpod() else "production",
        before_send=before_send,
        ignore_errors=[KeyboardInterrupt, FileNotFoundError],
    )
    sentry_sdk.set_user({"id": SETTINGS["uuid"]})  # SHA-256 anonymized UUID hash


class JSONDict(dict):
    """A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock and handles JSON serialization of Path objects.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Load the data from the JSON file into the dictionary.
        _save: Save the current state of the dictionary to the JSON file.
        __setitem__: Store a key-value pair and persist it to disk.
        __delitem__: Remove an item and update the persistent storage.
        update: Update the dictionary and persist changes.
        clear: Clear all entries and update the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """

    def __init__(self, file_path: str | Path = "data.json"):
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        super().__init__()
        self.file_path = Path(file_path)
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load the data from the JSON file into the dictionary."""
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    # Use the base dict update to avoid persisting during reads
                    super().update(json.load(f))
        except json.JSONDecodeError:
            LOGGER.warning(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            LOGGER.error(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        """Save the current state of the dictionary to the JSON file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=2, default=self._json_default)
        except Exception as e:
            LOGGER.error(f"Error writing to {self.file_path}: {e}")

    @staticmethod
    def _json_default(obj):
        """Handle JSON serialization of Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        """Store a key-value pair and persist to disk."""
        with self.lock:
            super().__setitem__(key, value)
            self._save()

    def __delitem__(self, key):
        """Remove an item and update the persistent storage."""
        with self.lock:
            super().__delitem__(key)
            self._save()

    def __str__(self):
        """Return a pretty-printed JSON string representation of the dictionary."""
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()


class SettingsManager(JSONDict):
    """SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings. The settings
    include directories for datasets, weights, and runs, as well as various integration flags.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validate the current settings and reset if necessary.
        update: Update settings, validating keys and types.
        reset: Reset the settings to default and save them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """

    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        """Initialize the SettingsManager with default settings and load user settings."""
        import hashlib
        import uuid

        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        root = GIT.root or Path()
        datasets_root = (root.parent if GIT.root and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            "settings_version": version,  # Settings schema version
            "datasets_dir": str(datasets_root / "datasets"),  # Datasets directory
            "weights_dir": str(root / "weights"),  # Model weights directory
            "runs_dir": str(root / "runs"),  # Experiment runs directory
            "uuid": hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
            "sync": True,  # Enable synchronization
            "api_key": "",  # Ultralytics API Key
            "openai_api_key": "",  # OpenAI API Key
            "clearml": True,  # ClearML integration
            "comet": True,  # Comet integration
            "dvc": True,  # DVC integration
            "hub": True,  # Ultralytics HUB integration
            "mlflow": True,  # MLflow integration
            "neptune": True,  # Neptune integration
            "raytune": True,  # Ray Tune integration
            "tensorboard": False,  # TensorBoard logging
            "wandb": False,  # Weights & Biases logging
            "vscode_msg": True,  # VSCode message
            "openvino_msg": True,  # OpenVINO export on Intel CPU message
        }

        self.help_msg = (
            f"\nView Ultralytics Settings with 'yolo settings' or at '{self.file}'"
            "\nUpdate Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. "
            "For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."
        )

        with torch_distributed_zero_first(LOCAL_RANK):
            super().__init__(self.file)

            if not self.file.exists() or not self:  # Check if file doesn't exist or is empty
                LOGGER.info(f"Creating new Ultralytics Settings v{version} file ✅ {self.help_msg}")
                self.reset()

            self._validate_settings()

    def _validate_settings(self):
        """Validate the current settings and reset if necessary."""
        correct_keys = frozenset(self.keys()) == frozenset(self.defaults.keys())
        correct_types = all(isinstance(self.get(k), type(v)) for k, v in self.defaults.items())
        correct_version = self.get("settings_version", "") == self.version

        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                "Ultralytics settings reset to default values. This may be due to a possible problem "
                f"with your settings or a recent ultralytics package update. {self.help_msg}"
            )
            self.reset()

        if self.get("datasets_dir") == self.get("runs_dir"):
            LOGGER.warning(
                f"Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' "
                f"must be different than 'runs_dir: {self.get('runs_dir')}'. "
                f"Please change one to avoid possible issues during training. {self.help_msg}"
            )

    def __setitem__(self, key, value):
        """Update one key: value pair."""
        self.update({key: value})

    def update(self, *args, **kwargs):
        """Update settings, validating keys and types."""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise KeyError(f"No Ultralytics setting '{k}'. {self.help_msg}")
            t = type(self.defaults[k])
            if not isinstance(v, t):
                raise TypeError(
                    f"Ultralytics setting '{k}' must be '{t.__name__}' type, not '{type(v).__name__}'. {self.help_msg}"
                )
        super().update(*args, **kwargs)

    def reset(self):
        """Reset the settings to default and save them."""
        self.clear()
        self.update(self.defaults)


def deprecation_warn(arg, new_arg=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    msg = f"'{arg}' is deprecated and will be removed in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return unquote(url).split("?", 1)[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """Display a message to install Ultralytics-Snippets for VS Code if not already installed."""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # file tracks uninstalled extensions, while source directory remains
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"


# Run below code on utils init ------------------------------------------------------------------------------------

# Check first-install steps
PREFIX = colorstr("Ultralytics: ")
SETTINGS = SettingsManager()  # initialize settings
PERSISTENT_CACHE = JSONDict(USER_CONFIG_DIR / "persistent_cache.json")  # initialize persistent cache
DATASETS_DIR = Path(SETTINGS["datasets_dir"])  # global datasets directory
WEIGHTS_DIR = Path(SETTINGS["weights_dir"])  # global weights directory
RUNS_DIR = Path(SETTINGS["runs_dir"])  # global runs directory
ENVIRONMENT = (
    "Colab"
    if IS_COLAB
    else "Kaggle"
    if IS_KAGGLE
    else "Jupyter"
    if IS_JUPYTER
    else "Docker"
    if IS_DOCKER
    else platform.system()
)
TESTS_RUNNING = is_pytest_running() or is_github_action_running()
set_sentry()

# Apply monkey patches
torch.save = torch_save
if WINDOWS:
    # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
