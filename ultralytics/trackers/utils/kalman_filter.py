"""
卡尔曼滤波器模块

此模块实现了用于目标追踪的卡尔曼滤波器，提供状态预测和更新功能。
卡尔曼滤波是一种最优递归数据处理算法，广泛应用于目标追踪、导航等领域。

主要类:
    - KalmanFilterXYAH: 使用 XYAH 格式的卡尔曼滤波器（中心点 + 纵横比 + 高度）
    - KalmanFilterXYWH: 使用 XYWH 格式的卡尔曼滤波器（中心点 + 宽度 + 高度）

核心功能:
    - initiate: 从检测初始化追踪状态
    - predict: 预测下一时刻的状态（运动模型）
    - update: 用新观测更新状态（测量更新）
    - project: 将状态投影到观测空间
    - gating_distance: 计算门限距离（用于数据关联）

数学模型:
    状态空间: [x, y, a/w, h, vx, vy, va/vw, vh]
        - (x, y): 边界框中心坐标
        - a: 纵横比 (XYAH) 或 w: 宽度 (XYWH)
        - h: 高度
        - v*: 对应的速度分量

    运动模型: 恒速模型 (Constant Velocity Model)
        x(t+1) = F * x(t) + w(t)
        其中 F 是状态转移矩阵，w(t) 是过程噪声

    观测模型: 线性观测模型
        z(t) = H * x(t) + v(t)
        其中 H 是观测矩阵，v(t) 是观测噪声

参考资料:
    - Kalman Filter: https://en.wikipedia.org/wiki/Kalman_filter
    - SORT: https://arxiv.org/abs/1602.00763
"""

import numpy as np  # 数值计算库
import scipy.linalg  # 科学计算库（用于 Cholesky 分解等）


class KalmanFilterXYAH:
    """A KalmanFilterXYAH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a simple Kalman filter for tracking bounding boxes in image space. The 8-dimensional state space (x, y,
    a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect ratio a, height h, and their
    respective velocities. Object motion follows a constant velocity model, and bounding box location (x, y, a, h) is
    taken as a direct observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step (vectorized version).
        update: Run the Kalman filter correction step.
        gating_distance: Compute the gating distance between state distribution and measurements.

    Examples:
        Initialize the Kalman filter and create a track from a measurement
        >>> kf = KalmanFilterXYAH()
        >>> measurement = np.array([100, 200, 1.5, 50])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights.

        初始化卡尔曼滤波器的模型矩阵和不确定性权重。

        The Kalman filter is initialized with an 8-dimensional state space (x, y, a, h, vx, vy, va, vh), where (x, y)
        represents the bounding box center position, 'a' is the aspect ratio, 'h' is the height, and their respective
        velocities are (vx, vy, va, vh). The filter uses a constant velocity model for object motion and a linear
        observation model for bounding box location.

        卡尔曼滤波器使用 8 维状态空间 (x, y, a, h, vx, vy, va, vh)：
            - (x, y): 边界框中心位置
            - a: 纵横比（宽度/高度）
            - h: 高度
            - (vx, vy, va, vh): 对应的速度分量

        采用恒速运动模型和线性观测模型。
        """
        ndim, dt = 4, 1.0  # 状态维度和时间步长（假设帧率恒定）

        # 创建状态转移矩阵 F (8×8)
        # Create Kalman filter model matrices
        # F = [I  dt*I]  其中 I 是单位矩阵
        #     [0   I  ]
        # 这表示：位置 = 位置 + 速度 * dt，速度保持不变
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt  # 设置位置-速度耦合项

        # 创建观测矩阵 H (4×8)
        # H = [I  0]  只观测位置部分，不直接观测速度
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 运动和观测不确定性权重（相对于当前状态估计）
        # Motion and observation uncertainty are chosen relative to the current state estimate
        # 这些权重用于构造过程噪声和观测噪声的协方差矩阵
        self._std_weight_position = 1.0 / 20  # 位置标准差权重（1/20 = 5%）
        self._std_weight_velocity = 1.0 / 160  # 速度标准差权重（1/160 = 0.625%）

    def initiate(self, measurement: np.ndarray):
        """Create a track from an unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            mean (np.ndarray): Mean vector (8-dimensional) of the new track. Unobserved velocities are initialized to 0
                mean.
            covariance (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> measurement = np.array([100, 50, 1.5, 200])
            >>> mean, covariance = kf.initiate(measurement)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        执行卡尔曼滤波器的预测步骤，根据运动模型预测下一时刻的状态。

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
                前一时刻的 8 维状态均值向量
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time
                step.
                前一时刻的 8×8 协方差矩阵

        Returns:
            mean (np.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
                预测的状态均值向量
            covariance (np.ndarray): Covariance matrix of the predicted state.
                预测的协方差矩阵

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)

        数学公式:
            预测步骤（先验估计）：
                mean' = F * mean
                covariance' = F * covariance * F^T + Q
            其中 Q 是过程噪声协方差矩阵（motion_cov）
        """
        # 构造过程噪声协方差矩阵 Q
        # 位置噪声标准差（与高度成正比）
        std_pos = [
            self._std_weight_position * mean[3],  # x 方向
            self._std_weight_position * mean[3],  # y 方向
            1e-2,  # 纵横比（固定小值）
            self._std_weight_position * mean[3],  # 高度
        ]
        # 速度噪声标准差（与高度成正比）
        std_vel = [
            self._std_weight_velocity * mean[3],  # vx
            self._std_weight_velocity * mean[3],  # vy
            1e-5,  # va（固定极小值）
            self._std_weight_velocity * mean[3],  # vh
        ]
        # 构造对角协方差矩阵（假设各维度独立）
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 预测状态均值：mean' = F * mean
        # 使用转置是因为 NumPy 的 dot 约定
        mean = np.dot(mean, self._motion_mat.T)

        # 预测协方差：covariance' = F * covariance * F^T + Q
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            mean (np.ndarray): Projected mean of the given state estimate.
            covariance (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_covariance = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step for multiple object states (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.random.rand(10, 8)  # 10 object states
            >>> covariance = np.random.rand(10, 8, 8)  # Covariance matrices for 10 object states
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        执行卡尔曼滤波器的更新步骤，用新的观测值校正预测状态。

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
                预测的状态均值向量（8 维）
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
                状态协方差矩阵（8×8）
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.
                4 维观测向量 (x, y, a, h)，其中 (x,y) 是中心位置，a 是纵横比，h 是高度

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
                校正后的状态均值
            new_covariance (np.ndarray): Measurement-corrected state covariance.
                校正后的状态协方差

        Examples:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([1, 1, 1, 1])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)

        数学公式:
            更新步骤（后验估计）：
                K = P * H^T * (H * P * H^T + R)^(-1)  [卡尔曼增益]
                mean' = mean + K * (z - H * mean)    [状态更新]
                P' = P - K * H * P                    [协方差更新]
            其中：
                K 是卡尔曼增益
                z 是观测值（measurement）
                R 是观测噪声协方差
        """
        # 将状态投影到观测空间
        projected_mean, projected_cov = self.project(mean, covariance)

        # 计算卡尔曼增益 K
        # 使用 Cholesky 分解求解线性方程组，数值稳定性更好
        # projected_cov = S = H * P * H^T + R
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # 求解 K^T = S^(-1) * (H * P)^T
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T

        # 计算新息（innovation）：观测值与预测观测值的差
        innovation = measurement - projected_mean

        # 更新状态均值：mean' = mean + K * innovation
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # 更新协方差：P' = P - K * S * K^T
        # 这是 Joseph 形式的简化版本
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square
        distribution has 4 degrees of freedom, otherwise 2.

        Args:
            mean (np.ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (np.ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (np.ndarray): An (N, 4) matrix of N measurements, each in format (x, y, a, h) where (x, y) is
                the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to box center position
                only.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.

        Examples:
            Compute gating distance using Mahalanobis metric:
            >>> kf = KalmanFilterXYAH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurements = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
            >>> distances = kf.gating_distance(mean, covariance, measurements, only_position=False, metric="maha")
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError("Invalid distance metric")


class KalmanFilterXYWH(KalmanFilterXYAH):
    """A KalmanFilterXYWH class for tracking bounding boxes in image space using a Kalman filter.

    Implements a Kalman filter for tracking bounding boxes with state space (x, y, w, h, vx, vy, vw, vh), where (x, y)
    is the center position, w is the width, h is the height, and vx, vy, vw, vh are their respective velocities. The
    object motion follows a constant velocity model, and the bounding box location (x, y, w, h) is taken as a direct
    observation of the state space (linear observation model).

    Attributes:
        _motion_mat (np.ndarray): The motion matrix for the Kalman filter.
        _update_mat (np.ndarray): The update matrix for the Kalman filter.
        _std_weight_position (float): Standard deviation weight for position.
        _std_weight_velocity (float): Standard deviation weight for velocity.

    Methods:
        initiate: Create a track from an unassociated measurement.
        predict: Run the Kalman filter prediction step.
        project: Project the state distribution to measurement space.
        multi_predict: Run the Kalman filter prediction step in a vectorized manner.
        update: Run the Kalman filter correction step.

    Examples:
        Create a Kalman filter and initialize a track
        >>> kf = KalmanFilterXYWH()
        >>> measurement = np.array([100, 50, 20, 40])
        >>> mean, covariance = kf.initiate(measurement)
        >>> print(mean)
        >>> print(covariance)
    """

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and
                height.

        Returns:
            mean (np.ndarray): Mean vector (8 dimensional) of the new track. Unobserved velocities are initialized to 0
                mean.
            covariance (np.ndarray): Covariance matrix (8x8 dimensional) of the new track.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> measurement = np.array([100, 50, 20, 40])
            >>> mean, covariance = kf.initiate(measurement)
            >>> print(mean)
            [100.  50.  20.  40.   0.   0.   0.   0.]
            >>> print(covariance)
            [[ 4.  0.  0.  0.  0.  0.  0.  0.]
             [ 0.  4.  0.  0.  0.  0.  0.  0.]
             [ 0.  0.  4.  0.  0.  0.  0.  0.]
             [ 0.  0.  0.  4.  0.  0.  0.  0.]
             [ 0.  0.  0.  0.  0.25  0.  0.  0.]
             [ 0.  0.  0.  0.  0.  0.25  0.  0.]
             [ 0.  0.  0.  0.  0.  0.  0.25  0.]
             [ 0.  0.  0.  0.  0.  0.  0.  0.25]]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Args:
            mean (np.ndarray): The 8-dimensional mean vector of the object state at the previous time step.
            covariance (np.ndarray): The 8x8-dimensional covariance matrix of the object state at the previous time
                step.

        Returns:
            mean (np.ndarray): Mean vector of the predicted state. Unobserved velocities are initialized to 0 mean.
            covariance (np.ndarray): Covariance matrix of the predicted state.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> predicted_mean, predicted_covariance = kf.predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            mean (np.ndarray): Projected mean of the given state estimate.
            covariance (np.ndarray): Projected covariance matrix of the given state estimate.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> projected_mean, projected_cov = kf.project(mean, covariance)
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (np.ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            mean (np.ndarray): Mean matrix of the predicted states with shape (N, 8).
            covariance (np.ndarray): Covariance matrix of the predicted states with shape (N, 8, 8).

        Examples:
            >>> mean = np.random.rand(5, 8)  # 5 objects with 8-dimensional state vectors
            >>> covariance = np.random.rand(5, 8, 8)  # 5 objects with 8x8 covariance matrices
            >>> kf = KalmanFilterXYWH()
            >>> predicted_mean, predicted_covariance = kf.multi_predict(mean, covariance)
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8 dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            new_mean (np.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.

        Examples:
            >>> kf = KalmanFilterXYWH()
            >>> mean = np.array([0, 0, 1, 1, 0, 0, 0, 0])
            >>> covariance = np.eye(8)
            >>> measurement = np.array([0.5, 0.5, 1.2, 1.2])
            >>> new_mean, new_covariance = kf.update(mean, covariance, measurement)
        """
        return super().update(mean, covariance, measurement)
