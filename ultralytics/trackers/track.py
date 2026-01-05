"""
追踪器注册与回调模块

此模块提供了追踪器的注册机制和预测回调函数，将追踪功能集成到 YOLO 预测流程中。

主要功能:
    - 追踪器初始化：根据配置文件创建追踪器实例
    - 预测回调：在预测开始和结束时调用追踪器
    - 多视频流支持：为每个视频流维护独立的追踪器
    - ReID 特征提取：支持自动提取 ReID 特征用于 BOTSort

核心函数:
    - on_predict_start: 预测开始前初始化追踪器
    - on_predict_postprocess_end: 预测后处理结束后更新追踪
    - register_tracker: 注册追踪回调到模型

使用流程:
    1. 注册追踪器：register_tracker(model, persist=True)
    2. 预测时自动调用回调函数
    3. 追踪器自动更新并返回追踪结果
"""

from functools import partial  # 偏函数，用于绑定参数
from pathlib import Path 

import torch  

from ultralytics.utils import YAML, IterableSimpleNamespace  
from ultralytics.utils.checks import check_yaml  

from .bot_sort import BOTSORT  # BOTSort 追踪器
from .byte_tracker import BYTETracker  # ByteTrack 追踪器

# 追踪器类型映射字典：将配置中的字符串映射到追踪器类
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """

    在预测开始前初始化追踪器，为每个视频流创建独立的追踪器实例。

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
            预测器对象，将在其上初始化追踪器
        persist (bool, optional): Whether to persist the trackers if they already exist.
            是否持久化追踪器（如果已存在则不重新创建）

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    # 分类任务不支持追踪模式
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")

    # 如果追踪器已存在且要求持久化，直接返回
    if hasattr(predictor, "trackers") and persist:
        return

    # 加载追踪器配置文件（YAML 格式）
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    # 验证追踪器类型
    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    # 重置特征提取器（防止之前使用的残留）
    predictor._feats = None  # reset in case used earlier
    # 移除之前注册的钩子（如果存在）
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()

    # 如果是 BOTSort + ReID + 自动模型，需要特殊处理
    if cfg.tracker_type == "botsort" and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        # 检查模型是否支持自动 ReID 特征提取
        # 需要：1) PyTorch 模型 2) Detect 头 3) 非端到端模式
        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            # 如果不支持，使用外部分类模型
            cfg.model = "yolo11n-cls.pt"
        else:
            # 注册前向钩子，提取 Detect 层的输入特征
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                # 保存特征（展开为新列表避免前向传播时被修改）
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            # 在 Detect 层前注册钩子
            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    # 为每个视频流创建追踪器实例
    trackers = []
    for _ in range(predictor.dataset.bs):
        # 使用配置创建追踪器（假设帧率 30fps）
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        # 非流模式只需要一个追踪器
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes
            break
    predictor.trackers = trackers
    # 初始化视频路径列表，用于判断何时重置追踪器（处理新视频时）
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """

    在预测后处理结束后，用追踪器更新检测结果，添加追踪 ID。

    Args:
        predictor (object): The predictor object containing the predictions.
            包含预测结果的预测器对象
        persist (bool, optional): Whether to persist the trackers if they already exist.
            是否持久化追踪器（跨视频保持追踪状态）

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    # 判断任务类型和数据集模式
    is_obb = predictor.args.task == "obb"  # 是否为旋转框检测
    is_stream = predictor.dataset.mode == "stream"  # 是否为流模式

    # 遍历所有预测结果
    for i, result in enumerate(predictor.results):
        # 选择对应的追踪器（流模式每个流独立，否则共享）
        tracker = predictor.trackers[i if is_stream else 0]
        # 获取当前视频的保存路径
        vid_path = predictor.save_dir / Path(result.path).name

        # 如果是新视频且不持久化，重置追踪器
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()  # 清空所有追踪状态
            predictor.vid_path[i if is_stream else 0] = vid_path

        # 提取检测结果（OBB 或普通框）
        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        # 使用追踪器更新：输入检测结果、原图、特征（如果有）
        # 返回格式：[xyxy/xywha, track_id, score, cls, idx]
        tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))

        # 如果没有追踪结果，跳过
        if len(tracks) == 0:
            continue

        # 提取检测索引（最后一列），用于对结果重排序
        idx = tracks[:, -1].astype(int)
        # 根据追踪器返回的顺序重排结果
        predictor.results[i] = result[idx]

        # 更新结果对象的边界框（去掉最后的 idx 列）
        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """

    将追踪回调函数注册到模型，使模型在预测时自动调用追踪器。
    这是启用追踪功能的入口函数。

    Args:
        model (object): The model object to register tracking callbacks for.
            要注册追踪回调的模型对象
        persist (bool): Whether to persist the trackers if they already exist.
            是否持久化追踪器（跨视频保持追踪状态）

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)

    工作流程:
        1. 在预测开始时调用 on_predict_start 初始化追踪器
        2. 模型执行检测
        3. 在后处理结束后调用 on_predict_postprocess_end 更新追踪
    """
    # 注册 "预测开始" 回调：初始化追踪器
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    # 注册 "后处理结束" 回调：更新追踪结果
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
