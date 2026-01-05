# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics YOLO 主模块初始化文件

这个模块是 Ultralytics YOLO 库的入口点，提供了所有核心模型类的懒加载功能。
它定义了版本号、支持的模型类型，并通过 Python 的特殊方法实现了模块的动态导入。

主要功能:
    - 定义当前 YOLO 库的版本号
    - 设置环境变量以优化 CPU 使用
    - 提供懒加载机制，仅在首次访问时导入模型类
    - 导出公共 API 接口
    - 支持类型检查和 IDE 自动补全
"""

__version__ = "8.3.247"  # Ultralytics YOLO 当前版本号

# 导入标准库
import importlib  # 用于动态导入模块
import os  # 用于操作系统相关操作
from typing import TYPE_CHECKING  # 用于类型检查时的条件导入

# 设置环境变量（必须在其他导入之前设置）
if not os.environ.get("OMP_NUM_THREADS"):
    # 设置 OpenMP 线程数为 1，减少训练时的 CPU 使用率，避免线程竞争
    os.environ["OMP_NUM_THREADS"] = "1"

# 从工具模块导入资源路径和设置
from ultralytics.utils import ASSETS, SETTINGS  # ASSETS: 资源文件路径, SETTINGS: 全局设置
from ultralytics.utils.checks import check_yolo as checks  # 导入 YOLO 检查函数并重命名为 checks
from ultralytics.utils.downloads import download  # 导入下载功能函数

# 创建设置的别名，方便外部访问
settings = SETTINGS

# 定义所有支持的模型类型
# YOLO: 标准 YOLO 目标检测模型
# YOLOWorld: 开放词汇目标检测模型
# YOLOE: YOLO Efficient 高效模型
# NAS: Neural Architecture Search 神经架构搜索模型
# SAM: Segment Anything Model 通用分割模型
# FastSAM: 快速分割模型
# RTDETR: Real-Time DEtection TRansformer 实时检测 Transformer
MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

# 定义模块的公共接口（当使用 from ultralytics import * 时导出的内容）
__all__ = (
    "__version__",  # 版本号
    "ASSETS",  # 资源文件路径
    *MODELS,  # 所有模型类（解包元组）
    "checks",  # 检查函数
    "download",  # 下载函数
    "settings",  # 设置对象
)

# 类型检查条件块：仅在类型检查器运行时执行，不会在运行时导入
if TYPE_CHECKING:
    # 为类型检查器（如 mypy、PyCharm）启用类型提示
    # 直接导入所有模型类，但不影响运行时性能
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
    """
    动态属性访问方法，实现模型类的懒加载

    当用户尝试访问模块中不存在的属性时，Python 会调用此方法。
    这允许我们在首次访问模型类时才导入它，而不是在模块加载时就导入所有模型。
    这种懒加载机制可以显著减少导入时间和内存占用。

    Args:
        name (str): 要访问的属性名称

    Returns:
        模型类对象（如果 name 是有效的模型名称）

    Raises:
        AttributeError: 如果 name 不是有效的模型名称

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')  # 只有在这时才会真正导入 YOLO 类
    """
    if name in MODELS:
        # 如果请求的属性是支持的模型名称
        # 动态导入 ultralytics.models 模块，并从中获取对应的模型类
        return getattr(importlib.import_module("ultralytics.models"), name)

    # 如果不是支持的模型名称，抛出 AttributeError 异常
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    自定义目录列表方法，用于 IDE 自动补全和 dir() 函数

    此方法扩展了 dir() 函数的返回结果，使其包含懒加载的模型名称。
    这确保 IDE 可以正确地提供自动补全建议，即使这些模型类尚未被导入。

    Returns:
        list: 排序后的属性名称列表，包括全局变量和模型名称

    Examples:
        >>> import ultralytics
        >>> dir(ultralytics)  # 将显示所有可用的模型类和其他公共接口
    """
    # 合并模块的全局变量和 MODELS 元组中的名称
    # set(globals()): 获取当前模块的所有全局变量名称
    # set(MODELS): 获取所有支持的模型名称
    # sorted(): 对合并后的集合进行排序，使输出更加整洁
    return sorted(set(globals()) | set(MODELS))


# 主程序入口点
if __name__ == "__main__":
    # 当直接运行此文件时（而不是作为模块导入），打印版本号
    print(__version__)
