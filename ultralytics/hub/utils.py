# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics HUB 工具函数模块

该模块提供了与 Ultralytics HUB 交互所需的各种工具函数，包括网络请求处理、
Cookie 身份验证以及进度显示等功能。

主要功能:
    - 带进度条的 HTTP 请求
    - 智能重试机制的网络请求
    - Google Colab 环境下的 Cookie 身份验证
    - HUB API 和 Web 根地址配置

导出的函数:
    request_with_credentials: 在 Colab 中使用 Cookie 进行认证的 AJAX 请求
    requests_with_progress: 带进度条的 HTTP 请求
    smart_request: 带重试和超时的智能 HTTP 请求

导出的常量:
    HUB_API_ROOT: HUB API 根地址
    HUB_WEB_ROOT: HUB Web 根地址
    PREFIX: 日志消息前缀
    HELP_MSG: 帮助信息
"""

# 导入标准库
import os  # 操作系统接口
import threading  # 线程支持
import time  # 时间相关功能
from typing import Any  # 类型注解

# 导入 Ultralytics 工具
from ultralytics.utils import (
    IS_COLAB,  # 是否在 Google Colab 环境中
    LOGGER,  # 日志记录器
    TQDM,  # 进度条
    TryExcept,  # 异常处理装饰器
    colorstr,  # 彩色字符串
)

# HUB API 根地址（可通过环境变量 ULTRALYTICS_HUB_API 自定义）
HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
# HUB Web 根地址（可通过环境变量 ULTRALYTICS_HUB_WEB 自定义）
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

# 日志消息前缀（带颜色）
PREFIX = colorstr("Ultralytics HUB: ")
# 帮助信息
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> Any:
    """Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        (Any): The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not IS_COLAB:
        # 此函数仅支持在 Google Colab 环境中运行
        raise OSError("request_with_credentials() must run in a Colab environment")
    from google.colab import output  # Colab 输出工具
    from IPython import display  # IPython 显示工具

    # 在 Colab 中执行 JavaScript 代码，使用浏览器 Cookie 进行身份验证
    display.display(
        display.Javascript(
            f"""
            window._hub_tmp = new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("{url}", {{
                    method: 'POST',
                    credentials: 'include'  // 包含 Cookie
                }})
                    .then((response) => resolve(response.json()))
                    .then((json) => {{
                    clearTimeout(timeout);
                    }}).catch((err) => {{
                    clearTimeout(timeout);
                    reject(err);
                }});
            }});
            """
        )
    )
    # 从 JavaScript 获取返回值
    return output.eval_js("_hub_tmp")


def requests_with_progress(method: str, url: str, **kwargs):
    """Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (Any): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Notes:
        - If 'progress' is set to True, the progress bar will display the download progress for responses with a known
          content length.
        - If 'progress' is a number then progress bar will display assuming content length = progress.
    """
    import requests  # 作用域限定的导入，因为 requests 是慢速导入

    # 从 kwargs 中提取 progress 参数
    progress = kwargs.pop("progress", False)
    if not progress:
        # 如果不需要进度条，直接返回请求结果
        return requests.request(method, url, **kwargs)
    # 使用流式传输进行请求
    response = requests.request(method, url, stream=True, **kwargs)
    # 计算总大小：从响应头获取或使用 progress 参数值
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)
    try:
        # 创建进度条
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        # 逐块迭代响应内容
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))  # 更新进度
        pbar.close()  # 关闭进度条
    except requests.exceptions.ChunkedEncodingError:  # 避免 'Connection broken: IncompleteRead' 警告
        response.close()
    return response


def smart_request(
    method: str,
    url: str,
    retry: int = 3,
    timeout: int = 30,
    thread: bool = True,
    code: int = -1,
    verbose: bool = True,
    progress: bool = False,
    **kwargs,
):
    """Make an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying.
        thread (bool, optional): Whether to execute the request in a separate daemon thread.
        code (int, optional): An identifier for the request, used for logging purposes.
        verbose (bool, optional): A flag to determine whether to print out to console or not.
        progress (bool, optional): Whether to show a progress bar during the request.
        **kwargs (Any): Keyword arguments to be passed to the requests function specified in method.

    Returns:
        (requests.Response | None): The HTTP response object. If the request is executed in a separate thread, returns
            None.
    """
    retry_codes = (408, 500)  # 仅对这些状态码进行重试（请求超时、服务器错误）

    @TryExcept(verbose=verbose)  # 异常处理装饰器
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None  # 响应对象
        t0 = time.time()  # 记录初始时间
        for i in range(retry + 1):
            # 检查是否超时
            if (time.time() - t0) > timeout:
                break
            # 发送请求（可能带进度条）
            r = requests_with_progress(func_method, func_url, **func_kwargs)
            # 2xx 范围的状态码通常表示成功
            if r.status_code < 300:
                break
            # 尝试从响应中获取错误消息
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            # 首次尝试时记录详细信息
            if i == 0:
                if r.status_code in retry_codes:
                    # 可重试的错误
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:  # 速率限制
                    h = r.headers  # 响应头
                    m = (
                        f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). "
                        f"Please retry after {h['Retry-After']}s."
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                # 非重试状态码直接返回
                if r.status_code not in retry_codes:
                    return r
            # 指数退避：等待 2^i 秒后重试
            time.sleep(2**i)
        return r

    # 准备函数参数
    args = method, url
    kwargs["progress"] = progress
    if thread:
        # 在后台线程中执行请求
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        # 在当前线程中执行请求
        return func(*args, **kwargs)
