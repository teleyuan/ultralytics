"""
基础追踪类模块

此模块定义了 YOLO 目标追踪的基础类和数据结构，为所有追踪算法提供统一的接口和状态管理。

Module defines the base classes and structures for object tracking in YOLO.

主要类:
    - TrackState: 追踪状态枚举类，定义目标的生命周期状态
    - BaseTrack: 追踪基类，提供追踪所需的基础属性和方法

核心功能:
    - 统一的追踪状态管理（新建、追踪中、丢失、移除）
    - 唯一 ID 分配机制
    - 追踪历史记录
    - 特征存储与管理
    - 抽象方法定义（激活、预测、更新）
"""

from collections import OrderedDict  # 有序字典，用于存储追踪历史
from typing import Any  # 类型提示

import numpy as np  # 数值计算库


class TrackState:
    """
    追踪状态枚举类

    定义目标在追踪过程中可能处于的四种状态，用于管理追踪生命周期。

    Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New (int): State when the object is newly detected.
            新建状态 (0): 目标首次被检测到，尚未确认为稳定追踪
        Tracked (int): State when the object is successfully tracked in subsequent frames.
            追踪中 (1): 目标正在被成功追踪，持续更新状态
        Lost (int): State when the object is no longer tracked.
            丢失状态 (2): 目标暂时丢失，但保留追踪信息用于重新识别
        Removed (int): State when the object is removed from tracking.
            移除状态 (3): 目标已确认丢失或超出追踪范围，从追踪列表中删除

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        ...     print("Object is newly detected.")
    """

    New = 0  # 新检测到的目标
    Tracked = 1  # 正在追踪的目标
    Lost = 2  # 丢失的目标
    Removed = 3  # 已移除的目标


class BaseTrack:
    """
    目标追踪基类

    为所有追踪算法提供基础的属性和方法接口，是所有追踪器的抽象基类。
    实现了追踪 ID 管理、状态维护、历史记录等核心功能。

    Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count (int): Class-level counter for unique track IDs.
            类级别计数器，用于生成全局唯一的追踪 ID
        track_id (int): Unique identifier for the track.
            当前追踪目标的唯一标识符
        is_activated (bool): Flag indicating whether the track is currently active.
            追踪是否已激活的标志，True 表示追踪已确认并激活
        state (TrackState): Current state of the track.
            当前追踪状态（New/Tracked/Lost/Removed）
        history (OrderedDict): Ordered history of the track's states.
            有序的追踪历史记录字典
        features (list): List of features extracted from the object for tracking.
            目标的特征向量列表，用于外观匹配
        curr_feature (Any): The current feature of the object being tracked.
            当前帧的目标特征
        score (float): The confidence score of the tracking.
            追踪置信度分数
        start_frame (int): The frame number where tracking started.
            追踪开始的帧号
        frame_id (int): The most recent frame ID processed by the track.
            最近处理的帧 ID
        time_since_update (int): Frames passed since the last update.
            自上次更新以来经过的帧数
        location (tuple): The location of the object in the context of multi-camera tracking.
            多摄像头追踪中的目标位置

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
            返回目标最后被追踪的帧 ID
        next_id: Increments and returns the next global track ID.
            递增并返回下一个全局追踪 ID
        activate: Abstract method to activate the track.
            激活追踪（抽象方法，需子类实现）
        predict: Abstract method to predict the next state of the track.
            预测下一帧状态（抽象方法，需子类实现）
        update: Abstract method to update the track with new data.
            用新数据更新追踪（抽象方法，需子类实现）
        mark_lost: Marks the track as lost.
            标记追踪为丢失状态
        mark_removed: Marks the track as removed.
            标记追踪为移除状态
        reset_id: Resets the global track ID counter.
            重置全局追踪 ID 计数器

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """

    _count = 0  # 全局追踪 ID 计数器

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes.

        初始化一个新的追踪对象，设置所有基础属性的默认值。
        """
        self.track_id = 0  # 追踪 ID，初始化为 0
        self.is_activated = False  # 是否已激活，默认未激活
        self.state = TrackState.New  # 初始状态为新建
        self.history = OrderedDict()  # 追踪历史记录
        self.features = []  # 特征向量列表
        self.curr_feature = None  # 当前特征
        self.score = 0  # 置信度分数
        self.start_frame = 0  # 开始帧号
        self.frame_id = 0  # 当前帧 ID
        self.time_since_update = 0  # 自上次更新以来的帧数
        self.location = (np.inf, np.inf)  # 多摄像头追踪的位置信息

    @property
    def end_frame(self) -> int:
        """Return the ID of the most recent frame where the object was tracked.

        返回目标最后被追踪到的帧 ID。
        """
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Increment and return the next unique global track ID for object tracking.

        递增全局计数器并返回下一个唯一的追踪 ID。
        这是一个类方法，确保所有追踪对象的 ID 全局唯一。
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Activate the track with provided arguments, initializing necessary attributes for tracking.

        激活追踪对象的抽象方法，需要在子类中实现。
        激活过程通常包括初始化卡尔曼滤波器、分配追踪 ID 等操作。

        Args:
            *args: 激活所需的参数（由子类定义）
        """
        raise NotImplementedError

    def predict(self) -> None:
        """Predict the next state of the track based on the current state and tracking model.

        预测追踪对象下一帧状态的抽象方法，需要在子类中实现。
        通常使用卡尔曼滤波器进行状态预测。
        """
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the track with new observations and data, modifying its state and attributes accordingly.

        用新的观测数据更新追踪对象的抽象方法，需要在子类中实现。
        更新过程通常包括卡尔曼滤波器的更新步骤，以及属性的更新。

        Args:
            *args: 位置参数（由子类定义）
            **kwargs: 关键字参数（由子类定义）
        """
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark the track as lost by updating its state to TrackState.Lost.

        将追踪对象标记为丢失状态。
        当目标在连续帧中未被检测到时调用此方法。
        """
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark the track as removed by setting its state to TrackState.Removed.

        将追踪对象标记为移除状态。
        当目标长时间丢失或确认离开追踪范围时调用此方法。
        """
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to its initial value.

        重置全局追踪 ID 计数器为初始值 0。
        通常在开始处理新视频或重新初始化追踪器时调用。
        """
        BaseTrack._count = 0
