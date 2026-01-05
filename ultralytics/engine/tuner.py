"""
超参数调优模块

该模块提供了 Ultralytics YOLO 模型的超参数调优功能，支持目标检测、实例分割、
图像分类、姿态估计和多目标跟踪等任务。

超参数调优是一个系统性搜索最优超参数组合的过程，以获得最佳模型性能。
在 YOLO 等深度学习模型中，超参数的微小变化可能导致模型精度和效率的显著差异。

主要功能:
    - 使用进化算法搜索最优超参数
    - 支持本地 CSV 存储和分布式 MongoDB 协调
    - 自动记录和可视化调优结果
    - 支持自定义搜索空间

Examples:
    在 COCO8 数据集上调优 YOLO11n，图像尺寸 640，训练 10 轮，进行 300 次调优迭代:
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo11n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""

from __future__ import annotations  # 启用延迟类型注解评估

import gc  # 垃圾回收
import random  # 随机数生成
import shutil  # 文件操作工具
import subprocess  # 子进程管理
import time  # 时间相关函数
from datetime import datetime  # 日期时间处理

import numpy as np  # 数组和数值计算
import torch  # PyTorch 深度学习框架

# 配置和工具导入
from ultralytics.cfg import get_cfg, get_save_dir  # 配置管理和保存目录获取
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr  # 工具函数
from ultralytics.utils.checks import check_requirements  # 依赖检查
from ultralytics.utils.patches import torch_load  # 安全的模型加载
from ultralytics.utils.plotting import plot_tune_results  # 结果可视化


class Tuner:
    """用于 YOLO 模型超参数调优的类。

    该类通过根据搜索空间变异超参数并重新训练模型来评估其性能，在给定的迭代次数内进化
    YOLO 模型的超参数。支持本地 CSV 存储和分布式 MongoDB Atlas 协调，用于多机超参数优化。

    属性:
        space (dict[str, tuple]): 包含变异边界和缩放因子的超参数搜索空间。
        tune_dir (Path): 保存进化日志和结果的目录。
        tune_csv (Path): 保存进化日志的 CSV 文件路径。
        args (dict): 调优过程的配置参数。
        callbacks (list): 调优期间执行的回调函数。
        prefix (str): 日志消息的前缀字符串。
        mongodb (MongoClient): 用于分布式调优的可选 MongoDB 客户端。
        collection (Collection): 用于存储调优结果的 MongoDB 集合。

    方法:
        _mutate: 基于边界和缩放因子变异超参数。
        __call__: 在多次迭代中执行超参数进化。

    示例:
        在 COCO8 数据集上为 YOLO11n 调优超参数，图像大小 640，训练 10 个 epoch，进行 300 次调优迭代。
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     plots=False,
        >>>     save=False,
        >>>     val=False
        >>> )

        使用分布式 MongoDB Atlas 协调在多台机器上进行调优:
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     mongodb_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
        >>>     mongodb_db="ultralytics",
        >>>     mongodb_collection="tune_results"
        >>> )

        使用自定义搜索空间进行调优:
        >>> model.tune(space={"lr0": (1e-5, 1e-1), "momentum": (0.6, 0.98)})
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks: list | None = None):
        """初始化超参数调优器

        Args:
            args (dict): 超参数进化的配置字典
            _callbacks (list | None, optional): 调优过程中执行的回调函数列表
        """
        # 定义超参数搜索空间，格式为 key: (最小值, 最大值, 增益系数(可选))
        self.space = args.pop("space", None) or {
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # OneCycleLR 最终学习率 (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD 动量/Adam beta1 参数
            "weight_decay": (0.0, 0.001),  # 优化器权重衰减系数
            "warmup_epochs": (0.0, 5.0),  # 预热轮数（可以是小数）
            "warmup_momentum": (0.0, 0.95),  # 预热初始动量
            "box": (1.0, 20.0),  # 边界框损失增益
            "cls": (0.1, 4.0),  # 分类损失增益（随像素缩放）
            "dfl": (0.4, 6.0),  # DFL（Distribution Focal Loss）损失增益
            "hsv_h": (0.0, 0.1),  # HSV-色调增强（比例）
            "hsv_s": (0.0, 0.9),  # HSV-饱和度增强（比例）
            "hsv_v": (0.0, 0.9),  # HSV-明度增强（比例）
            "degrees": (0.0, 45.0),  # 图像旋转角度 (+/- deg)
            "translate": (0.0, 0.9),  # 图像平移比例 (+/- fraction)
            "scale": (0.0, 0.95),  # 图像缩放比例 (+/- gain)
            "shear": (0.0, 10.0),  # 图像剪切角度 (+/- deg)
            "perspective": (0.0, 0.001),  # 图像透视变换 (+/- fraction)，范围 0-0.001
            "flipud": (0.0, 1.0),  # 图像上下翻转概率
            "fliplr": (0.0, 1.0),  # 图像左右翻转概率
            "bgr": (0.0, 1.0),  # 图像通道 BGR 转换概率
            "mosaic": (0.0, 1.0),  # 马赛克增强概率
            "mixup": (0.0, 1.0),  # MixUp 增强概率
            "cutmix": (0.0, 1.0),  # CutMix 增强概率
            "copy_paste": (0.0, 1.0),  # 分割任务的复制粘贴增强概率
            "close_mosaic": (0.0, 10.0),  # 关闭马赛克增强的轮数
        }
        # 从配置中提取 MongoDB 相关参数
        mongodb_uri = args.pop("mongodb_uri", None)
        mongodb_db = args.pop("mongodb_db", "ultralytics")
        mongodb_collection = args.pop("mongodb_collection", "tuner_results")

        # 获取并设置配置
        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume  # 恢复训练时允许使用相同的 tune_dir
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")  # 获取调优结果保存目录
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # 重置参数以免影响训练
        self.tune_csv = self.tune_dir / "tune_results.csv"  # CSV 结果文件路径
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 设置回调函数
        self.prefix = colorstr("Tuner: ")  # 日志前缀（带颜色）
        callbacks.add_integration_callbacks(self)  # 添加集成回调

        # MongoDB Atlas 支持（可选，用于分布式调优）
        self.mongodb = None
        if mongodb_uri:
            self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _connect(self, uri: str = "mongodb+srv://username:password@cluster.mongodb.net/", max_retries: int = 3):
        """在连接失败时使用指数退避重试创建 MongoDB 客户端。

        参数:
            uri (str): 包含凭据和集群信息的 MongoDB 连接字符串。
            max_retries (int): 放弃前的最大连接尝试次数。

        返回:
            (MongoClient): 已连接的 MongoDB 客户端实例。
        """
        check_requirements("pymongo")

        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=40000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=30,
                    minPoolSize=3,
                    maxIdleTimeMS=60000,
                )
                client.admin.command("ping")  # Test connection
                LOGGER.info(f"{self.prefix}Connected to MongoDB Atlas (attempt {attempt + 1})")
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError):
                if attempt == max_retries - 1:
                    raise
                wait_time = 2**attempt
                LOGGER.warning(
                    f"{self.prefix}MongoDB connection failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def _init_mongodb(self, mongodb_uri="", mongodb_db="", mongodb_collection=""):
        """初始化用于分布式调优的 MongoDB 连接。

        连接到 MongoDB Atlas 以在多台机器上进行分布式超参数优化。每个工作进程
        将结果保存到共享集合，并从所有工作进程读取最新的最佳超参数用于进化。

        参数:
            mongodb_uri (str): MongoDB 连接字符串，例如 'mongodb+srv://username:password@cluster.mongodb.net/'。
            mongodb_db (str, optional): 数据库名称。
            mongodb_collection (str, optional): 集合名称。

        注意:
            - 创建适应度索引以快速查询最佳结果
            - 如果连接失败则回退到仅 CSV 模式
            - 使用连接池和重试逻辑以确保生产环境可靠性
        """
        self.mongodb = self._connect(mongodb_uri)
        self.collection = self.mongodb[mongodb_db][mongodb_collection]
        self.collection.create_index([("fitness", -1)], background=True)
        LOGGER.info(f"{self.prefix}Using MongoDB Atlas for distributed tuning")

    def _get_mongodb_results(self, n: int = 5) -> list:
        """从 MongoDB 获取按适应度排序的前 N 个结果。

        参数:
            n (int): 要检索的最佳结果数量。

        返回:
            (list[dict]): 包含适应度分数和超参数的结果文档列表。
        """
        try:
            return list(self.collection.find().sort("fitness", -1).limit(n))
        except Exception:
            return []

    def _save_to_mongodb(self, fitness: float, hyperparameters: dict[str, float], metrics: dict, iteration: int):
        """将结果以正确的类型转换保存到 MongoDB。

        参数:
            fitness (float): 使用这些超参数获得的适应度分数。
            hyperparameters (dict[str, float]): 超参数值字典。
            metrics (dict): 完整的训练指标字典(mAP、精确度、召回率、损失等)。
            iteration (int): 当前迭代编号。
        """
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
                    "metrics": metrics,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_csv(self):
        """将 MongoDB 结果同步到 CSV 以实现绘图兼容性。

        从 MongoDB 下载所有结果并按时间顺序将它们写入本地 CSV 文件。这使得
        现有的绘图函数能够无缝处理分布式 MongoDB 数据。
        """
        try:
            # Get all results from MongoDB
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return

            # Write to CSV
            headers = ",".join(["fitness", *list(self.space.keys())]) + "\n"
            with open(self.tune_csv, "w", encoding="utf-8") as f:
                f.write(headers)
                for result in all_results:
                    fitness = result["fitness"]
                    hyp_values = [result["hyperparameters"][k] for k in self.space.keys()]
                    log_row = [round(fitness, 5), *hyp_values]
                    f.write(",".join(map(str, log_row)) + "\n")

        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to CSV sync failed: {e}")

    @staticmethod
    def _crossover(x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
        """BLX-α 交叉操作，从前 k 个父代中进行混合 (x[:,0]=适应度, 其余=基因)

        使用 BLX-α（Blend Crossover Alpha）算法从多个优秀父代中生成新的超参数组合。

        Args:
            x: 父代矩阵，第一列是适应度，其余列是超参数值
            alpha: 交叉混合系数，控制搜索范围的扩展
            k: 参与交叉的父代数量

        Returns:
            新生成的超参数向量
        """
        k = min(k, len(x))  # 确保 k 不超过可用的父代数量
        # 计算适应度权重（偏移到 >0）；如果退化则回退到均匀分布
        weights = x[:, 0] - x[:, 0].min() + 1e-6
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        # 根据适应度权重随机选择 k 个父代
        idxs = random.choices(range(len(x)), weights=weights, k=k)
        parents_mat = np.stack([x[i][1:] for i in idxs], 0)  # (k, ng) 去除适应度列
        # 计算所有父代基因的最小值和最大值
        lo, hi = parents_mat.min(0), parents_mat.max(0)
        span = hi - lo
        # 在扩展的范围内均匀采样生成新基因
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = 9,
        mutation: float = 0.5,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        """基于 self.space 中指定的边界和缩放因子变异超参数

        使用进化算法中的交叉和变异操作生成新的超参数组合。
        优先从 MongoDB 读取历史最优结果，否则从 CSV 文件读取。

        Args:
            n (int): 考虑的最优父代数量
            mutation (float): 每次迭代中参数发生变异的概率
            sigma (float): 高斯随机数生成器的标准差

        Returns:
            (dict[str, float]): 包含变异后超参数的字典
        """
        x = None

        # 如果可用，优先尝试从 MongoDB 获取历史结果
        if self.mongodb:
            if results := self._get_mongodb_results(n):
                # MongoDB 已按适应度降序排序，results[0] 是最佳结果
                x = np.array([[r["fitness"]] + [r["hyperparameters"][k] for k in self.space.keys()] for r in results])
            elif self.collection.name in self.collection.database.list_collection_names():  # 调优器在其他地方启动
                x = np.array([[0.0] + [getattr(self.args, k) for k in self.space.keys()]])

        # 如果 MongoDB 不可用或为空，回退到 CSV 文件
        if x is None and self.tune_csv.exists():
            csv_data = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            if len(csv_data) > 0:
                fitness = csv_data[:, 0]  # 第一列是适应度
                order = np.argsort(-fitness)  # 按适应度降序排序
                x = csv_data[order][:n]  # 取前 n 个最优结果

        # 如果有历史数据则进行变异，否则使用默认值
        if x is not None:
            np.random.seed(int(time.time()))  # 设置随机种子
            ng = len(self.space)  # 超参数数量

            # 步骤 1: 交叉操作 - 从多个父代中混合生成新基因
            genes = self._crossover(x)

            # 步骤 2: 变异操作 - 添加随机扰动
            gains = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # 增益系数 0-1
            factors = np.ones(ng)  # 变异因子初始化为 1
            while np.all(factors == 1):  # 确保至少有一个参数发生变化（防止重复）
                mask = np.random.random(ng) < mutation  # 随机选择要变异的参数
                step = np.random.randn(ng) * (sigma * gains)  # 高斯扰动步长
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)  # 计算变异因子并限制范围
            hyp = {k: float(genes[i] * factors[i]) for i, k in enumerate(self.space.keys())}
        else:
            # 没有历史数据时使用默认配置
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # 将超参数限制在指定的边界范围内
        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)

        # 更新特定参数的类型（例如将 close_mosaic 转为整数）
        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])

        return hyp

    def __call__(self, model=None, iterations: int = 10, cleanup: bool = True):
        """执行超参数进化过程（当 Tuner 实例被调用时）

        该方法通过指定数量的迭代执行超参数调优，执行以下步骤:
        1. 同步 MongoDB 结果到 CSV（如果使用分布式模式）
        2. 使用最佳历史结果或默认值变异超参数
        3. 使用变异后的超参数训练 YOLO 模型
        4. 将适应度分数和超参数记录到 MongoDB 和/或 CSV
        5. 跟踪所有迭代中性能最佳的配置

        Args:
            model (Model | None, optional): 预初始化的 YOLO 模型用于训练
            iterations (int): 进化运行的代数
            cleanup (bool): 是否删除迭代权重以减少调优期间的存储空间
        """
        t0 = time.time()  # 记录开始时间
        best_save_dir, best_metrics = None, None  # 最佳模型的保存目录和指标
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)  # 创建权重保存目录

        # 启动时同步 MongoDB 到 CSV 以支持恢复逻辑
        if self.mongodb:
            self._sync_mongodb_to_csv()

        # 检查是否有历史记录以支持恢复训练
        start = 0
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            start = x.shape[0]  # 已完成的迭代次数
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")

        # 主调优循环
        for i in range(start, iterations):
            # 在前 300 次迭代中线性衰减 sigma 从 0.2 → 0.1
            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac

            # 变异超参数生成新的候选配置
            mutated_hyp = self._mutate(sigma=sigma_i)
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            # 准备训练参数
            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}  # 合并基础配置和变异超参数
            save_dir = get_save_dir(get_cfg(train_args))
            weights_dir = save_dir / "weights"

            try:
                # 在子进程中训练 YOLO 模型（避免数据加载器挂起问题）
                launch = [__import__("sys").executable, "-m", "ultralytics.cfg.__init__"]  # 解决 yolo 未找到的问题
                cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
                return_code = subprocess.run(cmd, check=True).returncode

                # 加载训练指标
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                metrics = torch_load(ckpt_file)["train_metrics"]
                assert return_code == 0, "training failed"

                # 清理内存
                time.sleep(1)
                gc.collect()  # 垃圾回收
                torch.cuda.empty_cache()  # 清空 CUDA 缓存

            except Exception as e:
                LOGGER.error(f"training failure for hyperparameter tuning iteration {i + 1}\n{e}")

            # 保存结果 - MongoDB 优先
            fitness = metrics.get("fitness", 0.0)  # 获取适应度分数
            if self.mongodb:
                # 使用 MongoDB 分布式存储
                self._save_to_mongodb(fitness, mutated_hyp, metrics, i + 1)
                self._sync_mongodb_to_csv()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    LOGGER.info(
                        f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                    )
                    break
            else:
                # 仅使用 CSV 存储（无 MongoDB）
                log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
                headers = "" if self.tune_csv.exists() else (",".join(["fitness", *list(self.space.keys())]) + "\n")
                with open(self.tune_csv, "a", encoding="utf-8") as f:
                    f.write(headers + ",".join(map(str, log_row)) + "\n")

            # 获取并更新最佳结果
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # 第一列是适应度
            best_idx = fitness.argmax()  # 最佳适应度的索引
            best_is_current = best_idx == i  # 当前迭代是否是最佳

            if best_is_current:
                # 当前迭代是最佳结果，保存权重
                best_save_dir = str(save_dir)
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup and best_save_dir:
                # 删除非最佳迭代的目录以减少存储空间
                shutil.rmtree(best_save_dir, ignore_errors=True)

            # 绘制调优结果图表
            plot_tune_results(str(self.tune_csv))

            # 保存并打印调优结果
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}"
            )
            LOGGER.info("\n" + header)

            # 保存最佳超参数到 YAML 文件
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            YAML.save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            YAML.print(self.tune_dir / "best_hyperparameters.yaml")
