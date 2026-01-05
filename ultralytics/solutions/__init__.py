"""
解决方案模块初始化文件

本模块导出了 Ultralytics YOLO 的所有实用解决方案类，这些类为各种计算机视觉应用场景提供了
开箱即用的功能实现。

主要解决方案:
    - AIGym: AI 健身房 - 姿态估计与运动计数
    - Analytics: 数据分析 - 可视化图表生成
    - DistanceCalculation: 距离计算 - 测量物体间距离
    - Heatmap: 热力图 - 目标运动热力分布
    - InstanceSegmentation: 实例分割 - 物体实例级分割
    - ObjectBlurrer: 物体模糊 - 隐私保护
    - ObjectCounter: 物体计数 - 进出区域计数
    - ObjectCropper: 物体裁剪 - 检测目标裁剪保存
    - ParkingManagement: 停车管理 - 车位占用检测
    - QueueManager: 队列管理 - 排队人数统计
    - RegionCounter: 区域计数 - 多区域目标计数
    - SecurityAlarm: 安全告警 - 异常检测报警
    - SpeedEstimator: 速度估计 - 物体运动速度计算
    - TrackZone: 追踪区域 - 区域内目标追踪
    - VisionEye: 视觉眼睛 - 视觉焦点映射
    - VisualAISearch: 视觉搜索 - 基于语义的图像检索
    - Inference: Streamlit 推理界面 - Web 应用
"""

from .ai_gym import AIGym
from .analytics import Analytics
from .distance_calculation import DistanceCalculation
from .heatmap import Heatmap
from .instance_segmentation import InstanceSegmentation
from .object_blurrer import ObjectBlurrer
from .object_counter import ObjectCounter
from .object_cropper import ObjectCropper
from .parking_management import ParkingManagement, ParkingPtsSelection
from .queue_management import QueueManager
from .region_counter import RegionCounter
from .security_alarm import SecurityAlarm
from .similarity_search import SearchApp, VisualAISearch
from .speed_estimation import SpeedEstimator
from .streamlit_inference import Inference
from .trackzone import TrackZone
from .vision_eye import VisionEye

__all__ = (
    "AIGym",
    "Analytics",
    "DistanceCalculation",
    "Heatmap",
    "Inference",
    "InstanceSegmentation",
    "ObjectBlurrer",
    "ObjectCounter",
    "ObjectCropper",
    "ParkingManagement",
    "ParkingPtsSelection",
    "QueueManager",
    "RegionCounter",
    "SearchApp",
    "SecurityAlarm",
    "SpeedEstimator",
    "TrackZone",
    "VisionEye",
    "VisualAISearch",
)
