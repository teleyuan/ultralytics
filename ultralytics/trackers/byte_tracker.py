"""
ByteTrack 多目标追踪算法实现

此模块实现了 ByteTrack 算法，这是一种简单、快速且强大的多目标追踪方法。
ByteTrack 的核心思想是利用低置信度检测框来恢复被遮挡的目标，提高追踪召回率。

主要特点:
    - 高性能：基于卡尔曼滤波的运动预测
    - 鲁棒性：通过二次关联恢复低置信度目标
    - 简洁性：不依赖复杂的 ReID 特征提取

核心类:
    - STrack: 单个追踪目标，包含卡尔曼滤波器和状态管理
    - BYTETracker: ByteTrack 追踪器主类，管理所有追踪目标

算法流程:
    1. 将检测结果分为高置信度和低置信度两组
    2. 用高置信度检测与已有追踪进行第一次关联
    3. 用低置信度检测与剩余追踪进行第二次关联
    4. 初始化新追踪并移除长时间丢失的追踪

参考文献:
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations  # 启用延迟类型注解

from typing import Any  # 类型提示

import numpy as np  # 数值计算

from ..utils import LOGGER  # 日志记录器
from ..utils.ops import xywh2ltwh  # 坐标转换工具：中心点格式转左上角格式
from .basetrack import BaseTrack, TrackState  # 基础追踪类和状态枚举
from .utils import matching  # 匹配算法模块
from .utils.kalman_filter import KalmanFilterXYAH  # 卡尔曼滤波器（XYAH 格式）


class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        multi_gmc: Update multiple track states using a homography matrix.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)
                is the center, (w, h) are width and height, and `idx` is the detection index.
                边界框坐标，格式为 (中心x, 中心y, 宽度, 高度, 索引) 或带角度的 OBB 格式
            score (float): Confidence score of the detection.
                检测的置信度分数
            cls (Any): Class label for the detected object.
                检测目标的类别标签
        """
        super().__init__()
        # xywh+idx or xywha+idx
        # 验证输入格式：5个值（普通框）或6个值（带角度的 OBB 框）
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        # 将中心点格式转换为左上角格式 (top-left x, top-left y, width, height)
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        # 初始化卡尔曼滤波器为 None，稍后在 activate 时初始化
        self.kalman_filter = None
        # 卡尔曼滤波器的均值和协方差矩阵
        self.mean, self.covariance = None, None
        # 追踪是否已激活的标志
        self.is_activated = False

        # 保存检测属性
        self.score = score  # 置信度分数
        self.tracklet_len = 0  # 追踪长度（成功追踪的帧数）
        self.cls = cls  # 类别标签
        self.idx = xywh[-1]  # 检测索引
        self.angle = xywh[4] if len(xywh) == 6 else None  # 旋转角度（OBB 专用）

    def predict(self):
        """
        使用卡尔曼滤波器预测目标在下一帧的状态（位置和速度）。
        如果目标不在 Tracked 状态，则将纵横比变化速度设为 0。
        """
        mean_state = self.mean.copy()
        # 如果追踪状态不是 Tracked（即 Lost 或其他状态），将纵横比变化速度设为 0
        # mean_state[7] 对应 XYAH 格式中的 vh（纵横比变化速度）
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        # 执行卡尔曼滤波的预测步骤，更新均值和协方差
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """
        批量预测多个追踪目标的状态，提高计算效率。
        使用共享的卡尔曼滤波器对所有追踪进行向量化预测。
        """
        if len(stracks) <= 0:
            return
        # 收集所有追踪的均值和协方差矩阵
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # 对于非 Tracked 状态的追踪，将纵横比变化速度设为 0
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        # 使用共享卡尔曼滤波器进行批量预测（向量化操作，高效）
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        # 将预测结果更新回各个追踪对象
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks.

        使用单应性矩阵（Homography）批量更新多个追踪的位置和协方差。
        这用于补偿摄像头运动（GMC - Global Motion Compensation）。

        Args:
            stracks: 要更新的追踪列表
            H: 2x3 的仿射变换矩阵，默认为单位矩阵（无变换）
        """
        if stracks:
            # 收集所有追踪的均值和协方差
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 提取旋转部分（2x2）和平移部分
            R = H[:2, :2]
            # 构造 8x8 的旋转矩阵（用于 8 维状态空间）
            # 使用 Kronecker 积将 2x2 旋转矩阵扩展到 8x8
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]  # 平移向量

            # 对每个追踪应用变换
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # 应用旋转变换到状态向量
                mean = R8x8.dot(mean)
                # 应用平移到位置部分
                mean[:2] += t
                # 应用旋转变换到协方差矩阵
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                # 更新追踪对象的状态
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data and update its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    filtering for predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
        init_track: Initialize object tracking with detections.
        get_dists: Calculate the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
        remove_duplicate_stracks: Remove duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects.

        使用新检测结果更新追踪器，返回当前所有追踪目标的列表。
        这是 ByteTrack 算法的核心方法，实现了多阶段的数据关联策略。

        Args:
            results: 检测结果，包含边界框、置信度和类别
            img: 原始图像（可选，用于 GMC）
            feats: 特征向量（可选，用于外观匹配）

        Returns:
            np.ndarray: 追踪结果数组，每行包含 [xyxy/xywha, track_id, score, cls, idx]
        """
        self.frame_id += 1  # 增加帧计数器
        # 初始化本帧的追踪状态列表
        activated_stracks = []  # 新激活的追踪
        refind_stracks = []  # 重新找到的追踪（从丢失状态恢复）
        lost_stracks = []  # 新丢失的追踪
        removed_stracks = []  # 要移除的追踪

        # ===== Step 1: 将检测结果按置信度分为高低两组 =====
        scores = results.conf
        # 高置信度检测（大于等于 track_high_thresh，通常 0.5-0.6）
        remain_inds = scores >= self.args.track_high_thresh
        # 低置信度检测（大于 track_low_thresh 但小于 track_high_thresh）
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        # 低置信度检测索引：同时满足 > track_low_thresh 和 < track_high_thresh
        inds_second = inds_low & inds_high
        results_second = results[inds_second]  # 第二次关联用的低置信度检测
        results = results[remain_inds]  # 第一次关联用的高置信度检测
        # 准备特征向量（如果提供）
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        # 将高置信度检测初始化为 STrack 对象
        detections = self.init_track(results, feats_keep)

        # ===== 分类现有追踪：未确认的和已确认的 =====
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []  # 未确认的追踪（首次出现）
        tracked_stracks = []  # type: list[STrack]  # 已确认的追踪
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ===== Step 2: 第一次关联（高置信度检测 vs 已有追踪） =====
        # Step 2: First association, with high score detection boxes
        # 合并已追踪目标和丢失目标作为候选池
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # 使用卡尔曼滤波预测所有追踪目标的当前位置
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        # 如果有 GMC（全局运动补偿）模块，则补偿摄像头运动
        if hasattr(self, "gmc") and img is not None:
            # use try-except here to bypass errors from gmc module
            try:
                # 计算帧间的仿射变换矩阵
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                # 如果 GMC 失败，使用单位矩阵（无变换）
                warp = np.eye(2, 3)
            # 对所有追踪应用 GMC 变换
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 计算追踪与检测之间的距离（IoU 距离，可选融合置信度）
        dists = self.get_dists(strack_pool, detections)
        # 使用匈牙利算法进行线性分配，得到匹配对
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 处理匹配成功的追踪
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 如果追踪已激活，更新状态
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # 如果是丢失状态，重新激活
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ===== Step 3: 第二次关联（低置信度检测 vs 剩余追踪） =====
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        # 将低置信度检测初始化为 STrack 对象
        detections_second = self.init_track(results_second, feats_second)
        # 从第一次关联未匹配的追踪中，筛选出仍在 Tracked 状态的
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO: consider fusing scores or appearance features for second association.
        # 计算 IoU 距离（低置信度检测与剩余追踪）
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 使用较低的阈值（0.5）进行第二次关联
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 处理第二次关联的匹配结果
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 将第二次关联仍未匹配的追踪标记为丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ===== 处理未确认的追踪（首次出现的目标） =====
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        # 从第一次关联未匹配的高置信度检测中筛选
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        # 使用较高阈值（0.7）匹配未确认追踪
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        # 未确认追踪如果未匹配，直接移除
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # ===== Step 4: 初始化新追踪 =====
        # Step 4: Init new stracks
        # 对于仍未匹配的高置信度检测，初始化为新追踪
        for inew in u_detection:
            track = detections[inew]
            # 检查置信度是否达到新追踪阈值
            if track.score < self.args.new_track_thresh:
                continue
            # 激活新追踪
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # ===== Step 5: 更新追踪器状态 =====
        # Step 5: Update state
        # 移除长时间丢失的追踪
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 更新追踪列表：保留 Tracked 状态的追踪
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并新激活和重新找到的追踪
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        # 更新丢失列表：移除已重新追踪的目标
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        # 移除重复的追踪（基于 IoU）
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # 更新移除列表并限制大小
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

        # 返回所有已激活追踪的结果
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
