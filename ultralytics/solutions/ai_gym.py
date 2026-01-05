from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """
    AI健身房(AIGym)类：基于姿态估计监控实时视频流中的健身动作

    该类继承自BaseSolution，使用YOLO姿态估计模型监控健身训练。它根据预定义的上下位置角度阈值
    追踪和计数运动重复次数。主要应用于健身房中的健身动作自动计数和姿态分析。

    属性:
        states (dict[int, dict[str, float | int | str]]): 每个追踪目标的角度、重复次数和阶段状态
        up_angle (float): 判定运动"上"位置的角度阈值
        down_angle (float): 判定运动"下"位置的角度阈值
        kpts (list[int]): 用于角度计算的关键点索引列表

    方法:
        process: 处理一帧图像以检测姿态、计算角度并统计重复次数

    使用示例:
        >>> gym = AIGym(model="yolo11n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("处理后的图像", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化AIGym类，用于使用姿态估计和预定义角度监控健身训练

        Args:
            **kwargs (Any): 传递给父类构造函数的关键字参数，包括:
                - model (str): 模型名称或路径，默认为 "yolo11n-pose.pt"
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})  # 存储计数、角度和阶段的字典

        # 从配置中一次性提取详细信息供后续使用
        self.up_angle = float(self.CFG["up_angle"])  # 预定义的"上"姿态角度阈值
        self.down_angle = float(self.CFG["down_angle"])  # 预定义的"下"姿态角度阈值
        self.kpts = self.CFG["kpts"]  # 用户选择的健身动作关键点，供后续使用

    def process(self, im0) -> SolutionResults:
        """
        使用Ultralytics YOLO姿态模型监控健身训练

        该函数处理输入图像以追踪和分析人体姿态，用于健身监控。使用YOLO姿态模型检测关键点，
        估算角度，并根据预定义的角度阈值计数重复次数。

        处理流程：
        1. 提取追踪目标和关键点
        2. 计算关键点之间的角度
        3. 根据角度判断运动阶段（上/下）
        4. 统计完成的重复次数
        5. 绘制关键点、角度、计数和阶段信息

        Args:
            im0 (np.ndarray): 待处理的输入图像

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的标注图像
                - workout_count: 每个人完成的重复次数列表
                - workout_stage: 每个人当前的运动阶段列表
                - workout_angle: 每个人当前的角度列表
                - total_tracks: 追踪的总人数

        使用示例:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        self.extract_tracks(im0)  # 提取轨迹（边界框、类别和掩码）

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]  # 获取状态详情
                # 获取关键点并估算角度
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # 基于角度阈值判定阶段和计数逻辑
                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

                # 显示角度、计数和阶段文本
                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],  # 显示的角度文本
                        count_text=state["count"],  # 健身动作计数文本
                        stage_text=state["stage"],  # 阶段位置文本
                        center_kpt=k[int(self.kpts[1])],  # 用于显示的中心关键点
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 显示输出图像（如果环境支持显示）

        # 返回SolutionResults对象
        return SolutionResults(
            plot_im=plot_im,
            workout_count=[v["count"] for v in self.states.values()],
            workout_stage=[v["stage"] for v in self.states.values()],
            workout_angle=[v["angle"] for v in self.states.values()],
            total_tracks=len(self.track_ids),
        )
