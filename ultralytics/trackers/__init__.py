"""
目标追踪模块初始化文件

此模块是 Ultralytics YOLO 目标追踪功能的入口点，提供了两种主流的多目标追踪算法:
    - ByteTrack: 基于卡尔曼滤波的高性能多目标追踪算法
    - BOTSort: 结合 ReID 和 GMC 的鲁棒性增强追踪算法

主要功能:
    - 导出核心追踪器类 (BYTETracker, BOTSORT)
    - 提供追踪器注册接口 (register_tracker)
    - 支持视频序列中的目标检测与跟踪

使用示例:
    >>> from ultralytics.trackers import BYTETracker
    >>> tracker = BYTETracker(args, frame_rate=30)
    >>> results = tracker.update(detections, img)
"""

# 导入追踪算法实现
from .bot_sort import BOTSORT  # BOTSort 追踪器
from .byte_tracker import BYTETracker  # ByteTrack 追踪器
from .track import register_tracker  # 追踪器注册函数

# 定义模块公共接口
__all__ = "BOTSORT", "BYTETracker", "register_tracker"  # allow simpler import
