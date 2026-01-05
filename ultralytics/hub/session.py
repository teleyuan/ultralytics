# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics HUB 训练会话管理模块

该模块负责管理与 Ultralytics HUB 的训练会话，包括模型创建、加载、上传以及指标跟踪。
它封装了与 HUB 平台交互的所有核心功能，使用户能够在云端训练 YOLO 模型。

主要功能:
    - 创建和加载 HUB 模型
    - 管理训练会话的生命周期
    - 上传训练指标和模型检查点
    - 处理网络请求的重试和超时
    - 支持断点续训
    - 进度跟踪和显示

典型使用流程:
    1. 创建或加载模型
    2. 开始训练会话
    3. 定期上传训练指标
    4. 上传模型检查点
    5. 完成训练并上传最终模型

Classes:
    HUBTrainingSession: HUB 训练会话管理类
"""

from __future__ import annotations  # 支持类型注解中的前向引用

# 导入标准库
import shutil  # 文件操作工具
import threading  # 线程支持
import time  # 时间相关功能
from http import HTTPStatus  # HTTP 状态码
from pathlib import Path  # 路径操作
from typing import Any  # 类型注解
from urllib.parse import parse_qs, urlparse  # URL 解析工具

# 导入 Ultralytics 核心组件
from ultralytics import __version__  # 版本号
from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX  # HUB 工具和常量
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, TQDM, checks, emojis  # 通用工具
from ultralytics.utils.errors import HUBModelError  # HUB 模型错误类

# 代理名称：用于标识客户端类型（Colab 或本地）
AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"


class HUBTrainingSession:
    """HUB training session for Ultralytics HUB YOLO models.

    This class encapsulates the functionality for interacting with Ultralytics HUB during model training, including
    model creation, metrics tracking, and checkpoint uploading.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (dict[str, int]): Rate limits for different API calls in seconds.
        timers (dict[str, Any]): Timers for rate limiting.
        metrics_queue (dict[str, Any]): Queue for the model's metrics.
        metrics_upload_failed_queue (dict[str, Any]): Queue for metrics that failed to upload.
        model (Any): Model data fetched from Ultralytics HUB.
        model_file (str): Path to the model file.
        train_args (dict[str, Any]): Arguments for training the model.
        client (Any): Client for interacting with Ultralytics HUB.
        filename (str): Filename of the model.

    Examples:
        Create a training session with a model URL
        >>> session = HUBTrainingSession("https://hub.ultralytics.com/models/example-model")
        >>> session.upload_metrics()
    """

    def __init__(self, identifier: str):
        """Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session. It can be a URL string or a
                model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
        from hub_sdk import HUBClient  # 导入 HUB 客户端

        # 设置速率限制（单位：秒）
        # metrics: 指标上传间隔，ckpt: 检查点上传间隔，heartbeat: 心跳间隔
        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}
        self.metrics_queue = {}  # 保存每个 epoch 的指标，直到上传
        self.metrics_upload_failed_queue = {}  # 保存上传失败的指标
        self.timers = {}  # 保存计时器（在 ultralytics/utils/callbacks/hub.py 中使用）
        self.model = None  # HUB 模型对象
        self.model_url = None  # 模型的 HUB URL
        self.model_file = None  # 模型文件路径
        self.train_args = None  # 训练参数

        # 解析输入标识符
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # 获取认证凭据
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None

        # 初始化 HUB 客户端
        self.client = HUBClient(credentials)

        # 加载模型
        try:
            if model_id:
                self.load_model(model_id)  # 加载现有模型
            else:
                self.model = self.client.model()  # 加载空模型
        except Exception:
            # 如果是 HUB 模型 URL 且用户未认证，提示登录
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and not self.client.authenticated:
                LOGGER.warning(
                    f"{PREFIX}Please log in using 'yolo login API_KEY'. "
                    "You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys."
                )

    @classmethod
    def create_session(cls, identifier: str, args: dict[str, Any] | None = None):
        """Create an authenticated HUBTrainingSession or return None.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
            args (dict[str, Any], optional): Arguments for creating a new model if identifier is not a HUB model URL.

        Returns:
            session (HUBTrainingSession | None): An authenticated session or None if creation fails.
        """
        try:
            session = cls(identifier)
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL
                session.create_model(args)
                assert session.model.id, "HUB model not loaded correctly"
            return session
        # PermissionError and ModuleNotFoundError indicate hub-sdk not installed
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id: str):
        """Load an existing model from Ultralytics HUB using the provided model identifier.

        Args:
            model_id (str): The identifier of the model to load.

        Raises:
            ValueError: If the specified HUB model does not exist.
        """
        # 从 HUB 客户端加载模型
        self.model = self.client.model(model_id)
        if not self.model.data:  # 模型不存在
            raise ValueError(emojis("❌ The specified HUB model does not exist"))  # TODO: 改进错误处理

        # 设置模型 URL
        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        if self.model.is_trained():
            # 如果模型已训练完成，下载最佳权重
            LOGGER.info(f"Loading trained HUB model {self.model_url} 🚀")
            url = self.model.get_weights_url("best")  # 获取带认证的下载 URL
            # 下载模型文件到本地
            self.model_file = checks.check_file(url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id)
            return

        # 设置训练参数并启动心跳，让 HUB 监控代理状态
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    def create_model(self, model_args: dict[str, Any]):
        """Initialize a HUB training session with the specified model arguments.

        Args:
            model_args (dict[str, Any]): Arguments for creating the model, including batch size, epochs, image size,
                etc.

        Returns:
            (None): If the model could not be created.
        """
        # 构建模型创建的负载数据
        payload = {
            "config": {
                "batchSize": model_args.get("batch", -1),  # 批次大小
                "epochs": model_args.get("epochs", 300),  # 训练轮数
                "imageSize": model_args.get("imgsz", 640),  # 图像尺寸
                "patience": model_args.get("patience", 100),  # 早停耐心值
                "device": str(model_args.get("device", "")),  # 设备（将 None 转为字符串）
                "cache": str(model_args.get("cache", "ram")),  # 缓存方式（将 True, False, None 转为字符串）
            },
            "dataset": {"name": model_args.get("data")},  # 数据集名称
            "lineage": {
                # 架构信息：从文件名中移除扩展名
                "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},
                "parent": {},  # 父模型信息
            },
            "meta": {"name": self.filename},  # 元数据：模型文件名
        }

        # 如果是预训练模型（.pt 文件），设置父模型名称
        if self.filename.endswith(".pt"):
            payload["lineage"]["parent"]["name"] = self.filename

        # 调用 HUB API 创建模型
        self.model.create_model(payload)

        # 如果模型创建失败
        # TODO: 改进错误处理
        if not self.model.id:
            return None

        # 设置模型 URL
        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"

        # 启动心跳，让 HUB 监控代理状态
        self.model.start_heartbeat(self.rate_limits["heartbeat"])

        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")

    @staticmethod
    def _parse_identifier(identifier: str):
        """Parse the given identifier to determine the type and extract relevant components.

        The method supports different identifier formats:
            - A HUB model URL https://hub.ultralytics.com/models/MODEL
            - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY
            - A local filename that ends with '.pt' or '.yaml'

        Args:
            identifier (str): The identifier string to be parsed.

        Returns:
            api_key (str | None): Extracted API key if present.
            model_id (str | None): Extracted model ID if present.
            filename (str | None): Extracted filename if present.

        Raises:
            HUBModelError: If the identifier format is not recognized.
        """
        # 初始化返回值
        api_key, model_id, filename = None, None, None
        # 如果标识符是本地文件（.pt 或 .yaml）
        if identifier.endswith((".pt", ".yaml")):
            filename = identifier
        # 如果标识符是 HUB 模型 URL
        elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            parsed_url = urlparse(identifier)  # 解析 URL
            model_id = Path(parsed_url.path).stem  # 提取模型 ID（处理可能的尾部斜杠）
            query_params = parse_qs(parsed_url.query)  # 解析查询参数，如 {"api_key": ["API_KEY_HERE"]}
            api_key = query_params.get("api_key", [None])[0]  # 提取 API 密钥
        else:
            # 无法识别的标识符格式
            raise HUBModelError(f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID")
        return api_key, model_id, filename

    def _set_train_args(self):
        """Initialize training arguments and create a model entry on the Ultralytics HUB.

        This method sets up training arguments based on the model's state and updates them with any additional arguments
        provided. It handles different states of the model, such as whether it's resumable, pretrained, or requires
        specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are
                issues with the provided training arguments.
        """
        if self.model.is_resumable():
            # 模型有已保存的权重，支持断点续训
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
            self.model_file = self.model.get_weights_url("last")  # 获取最后一次保存的权重
        else:
            # 模型没有保存的权重
            self.train_args = self.model.data.get("train_args")  # 获取训练参数（新响应格式）

            # 设置模型文件：预训练模型使用父模型权重，否则使用架构配置文件
            self.model_file = (
                self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()
            )

        if "data" not in self.train_args:
            # RF bug - 数据集有时未导出
            raise ValueError("Dataset may still be processing. Please wait a minute and try again.")

        # 检查并转换 YOLOv5 文件名为 YOLOv5u（如果需要）
        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)
        self.model_id = self.model.id  # 保存模型 ID

    def request_queue(
        self,
        request_func,
        retry: int = 3,
        timeout: int = 30,
        thread: bool = True,
        verbose: bool = True,
        progress_total: int | None = None,
        stream_response: bool | None = None,
        *args,
        **kwargs,
    ):
        """Execute request_func with retries, timeout handling, optional threading, and progress tracking.

        Args:
            request_func (callable): The function to execute.
            retry (int): Number of retry attempts.
            timeout (int): Maximum time to wait for the request to complete.
            thread (bool): Whether to run the request in a separate thread.
            verbose (bool): Whether to log detailed messages.
            progress_total (int, optional): Total size for progress tracking.
            stream_response (bool, optional): Whether to stream the response.
            *args (Any): Additional positional arguments for request_func.
            **kwargs (Any): Additional keyword arguments for request_func.

        Returns:
            (requests.Response | None): The response object if thread=False, otherwise None.
        """

        def retry_request():
            """Attempt to call request_func with retries, timeout, and optional threading."""
            t0 = time.time()  # Record the start time for the timeout
            response = None
            for i in range(retry + 1):
                if (time.time() - t0) > timeout:
                    LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")
                    break  # Timeout reached, exit loop

                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(f"{PREFIX}Received no response from the request. {HELP_MSG}")
                    time.sleep(2**i)  # Exponential backoff before retrying
                    continue  # Skip further processing and retry

                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)

                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    # if request related to metrics upload
                    if kwargs.get("metrics"):
                        self.metrics_upload_failed_queue = {}
                    return response  # Success, no need to retry

                if i == 0:
                    # Initial attempt, check status code and provide messages
                    message = self._get_failure_message(response, retry, timeout)

                    if verbose:
                        LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")

                if not self._should_retry(response.status_code):
                    LOGGER.warning(f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code}")
                    break  # Not an error that should be retried, exit loop

                time.sleep(2**i)  # Exponential backoff for retries

            # if request related to metrics upload and exceed retries
            if response is None and kwargs.get("metrics"):
                self.metrics_upload_failed_queue.update(kwargs.get("metrics"))

            return response

        if thread:
            # Start a new thread to run the retry_request function
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            # If running in the main thread, call retry_request directly
            return retry_request()

    @staticmethod
    def _should_retry(status_code: int) -> bool:
        """Determine if a request should be retried based on the HTTP status code."""
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes

    def _get_failure_message(self, response, retry: int, timeout: int) -> str:
        """Generate a retry message based on the response status code.

        Args:
            response (requests.Response): The HTTP response object.
            retry (int): The number of retry attempts allowed.
            timeout (int): The maximum timeout duration.

        Returns:
            (str): The retry message.
        """
        if self._should_retry(response.status_code):
            return f"Retrying {retry}x for {timeout}s." if retry else ""
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # rate limit
            headers = response.headers
            return (
                f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). "
                f"Please retry after {headers['Retry-After']}s."
            )
        else:
            try:
                return response.json().get("message", "No JSON message.")
            except AttributeError:
                return "Unable to read JSON."

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        # 在新线程中上传指标队列的副本
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        """Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        weights = Path(weights)
        if not weights.is_file():
            # 权重文件不存在
            last = weights.with_name(f"last{weights.suffix}")
            if final and last.is_file():
                # 如果是最终上传且 best.pt 不存在，但 last.pt 存在
                # 这种情况通常发生在 Google Colab 等临时环境中断点续训时
                LOGGER.warning(
                    f"{PREFIX} Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. "
                    "This often happens when resuming training in transient environments like Google Colab. "
                    "For more reliable training, consider using Ultralytics HUB Cloud. "
                    "Learn more at https://docs.ultralytics.com/hub/cloud-training."
                )
                shutil.copy(last, weights)  # 复制 last.pt 为 best.pt
            else:
                LOGGER.warning(f"{PREFIX} Model upload issue. Missing model {weights}.")
                return

        # 上传模型到 HUB
        self.request_queue(
            self.model.upload_model,
            epoch=epoch,  # 当前 epoch
            weights=str(weights),  # 权重文件路径
            is_best=is_best,  # 是否为最佳模型
            map=map,  # 平均精度
            final=final,  # 是否为最终模型
            retry=10,  # 重试次数
            timeout=3600,  # 超时时间（秒）
            thread=not final,  # 非最终模型在后台线程上传
            progress_total=weights.stat().st_size if final else None,  # 仅最终模型显示进度
            stream_response=True,  # 流式响应
        )

    @staticmethod
    def _show_upload_progress(content_length: int, response) -> None:
        """Display a progress bar to track the upload progress of a file download."""
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    def _iterate_content(response) -> None:
        """Process the streamed HTTP response data."""
        for _ in response.iter_content(chunk_size=1024):
            pass  # Do nothing with data chunks
