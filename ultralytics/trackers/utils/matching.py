"""
目标匹配与数据关联模块

此模块提供了多目标追踪中的数据关联算法，用于将检测结果与已有追踪进行匹配。

主要功能:
    - linear_assignment: 线性分配算法（匈牙利算法）
    - iou_distance: 基于 IoU 的距离计算
    - embedding_distance: 基于外观特征的距离计算
    - fuse_score: 融合置信度分数

核心算法:
    1. 线性分配（匈牙利算法）：
       - 解决二分图最大权匹配问题
       - 输入成本矩阵，输出最优匹配对
       - 时间复杂度 O(n³)

    2. IoU 距离：
       - 计算追踪与检测的空间重叠度
       - 距离 = 1 - IoU，值越小表示越匹配

    3. 外观距离：
       - 计算 ReID 特征的余弦距离或欧氏距离
       - 用于长时间遮挡后的重识别

    4. 分数融合：
       - 将检测置信度融入距离计算
       - 提高高置信度检测的匹配权重

使用流程:
    1. 计算距离矩阵（IoU、外观等）
    2. 使用线性分配求解最优匹配
    3. 返回匹配对和未匹配的索引
"""

import numpy as np  # 数值计算
import scipy  # 科学计算库
from scipy.spatial.distance import cdist  # 距离计算函数

from ultralytics.utils.metrics import batch_probiou, bbox_ioa  # IoU 计算工具

# 尝试导入 LAP（Linear Assignment Problem）求解器
# LAP 库比 scipy 的实现更快
try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    使用线性分配算法（匈牙利算法）求解最优匹配问题。
    这是多目标追踪中数据关联的核心算法。

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
            成本矩阵，形状为 (N, M)，N 是追踪数，M 是检测数
        thresh (float): Threshold for considering an assignment valid.
            成本阈值，超过此值的匹配被认为无效
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.
            是否使用 LAP 库（默认 True，速度更快）

    Returns:
        matched_indices (list[list[int]] | np.ndarray): Matched indices of shape (K, 2), where K is the number of
            matches.
            匹配的索引对，形状为 (K, 2)，每行是 [追踪索引, 检测索引]
        unmatched_a (np.ndarray): Unmatched indices from the first set, with shape (L,).
            未匹配的追踪索引
        unmatched_b (np.ndarray): Unmatched indices from the second set, with shape (M,).
            未匹配的检测索引

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)

    算法说明:
        匈牙利算法求解二分图的最小权匹配问题：
        - 输入：成本矩阵 C[i,j] 表示追踪 i 与检测 j 的匹配成本
        - 输出：最优匹配方案，使总成本最小
        - 约束：每个追踪最多匹配一个检测，每个检测最多匹配一个追踪
    """
    # 处理空矩阵的情况
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # 使用 LAP 库的 LAPJV (Jonker-Volgenant) 算法
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        # extend_cost=True: 自动扩展矩阵为方阵
        # cost_limit: 超过此成本的匹配被拒绝
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        # x[i] = j 表示行 i 匹配到列 j，-1 表示未匹配
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]  # 未匹配的行（追踪）
        unmatched_b = np.where(y < 0)[0]  # 未匹配的列（检测）
    else:
        # 使用 SciPy 的线性分配实现
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        # 过滤掉成本超过阈值的匹配
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            # 如果没有有效匹配，所有行和列都是未匹配的
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            # 找出未匹配的行和列（使用集合差运算）
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks, where each track contains embedding features.
        detections (list[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
