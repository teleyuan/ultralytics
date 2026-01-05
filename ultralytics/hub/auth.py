# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics HUB 身份验证模块

该模块负责处理与 Ultralytics HUB 平台的身份验证相关功能，包括 API 密钥管理、
Cookie 身份验证（用于 Google Colab 环境）以及 HTTP 请求头生成。

主要功能:
    - 支持多种身份验证方式：API 密钥、浏览器 Cookie（Colab）、交互式输入
    - 自动保存和管理 API 密钥
    - 生成用于 API 请求的身份验证头
    - 验证用户凭据的有效性

典型使用场景:
    1. 直接使用 API 密钥进行身份验证
    2. 在 Google Colab 中使用浏览器 Cookie 进行身份验证
    3. 交互式提示用户输入 API 密钥

Classes:
    Auth: 身份验证管理类，处理所有身份验证相关操作
"""

# 导入 HUB 工具函数和常量
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, request_with_credentials

# 导入通用工具
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, emojis

# API 密钥获取页面的 URL
API_KEY_URL = f"{HUB_WEB_ROOT}/settings?tab=api+keys"


class Auth:
    """Manages authentication processes including API key handling, cookie-based authentication, and header generation.

    The class supports different methods of authentication:
    1. Directly using an API key.
    2. Authenticating using browser cookies (specifically in Google Colab).
    3. Prompting the user to enter an API key.

    Attributes:
        id_token (str | bool): Token used for identity verification, initialized as False.
        api_key (str | bool): API key for authentication, initialized as False.
        model_key (bool): Placeholder for model key, initialized as False.

    Methods:
        authenticate: Attempt to authenticate with the server using either id_token or API key.
        auth_with_cookies: Attempt to fetch authentication via cookies and set id_token.
        get_auth_header: Get the authentication header for making API requests.
        request_api_key: Prompt the user to input their API key.

    Examples:
        Initialize Auth with an API key
        >>> auth = Auth(api_key="your_api_key_here")

        Initialize Auth without API key (will prompt for input)
        >>> auth = Auth()
    """

    # 类级别的属性，用于存储身份验证信息
    id_token = api_key = model_key = False

    def __init__(self, api_key: str = "", verbose: bool = False):
        """Initialize Auth class and authenticate user.

        Handles API key validation, Google Colab authentication, and new key requests. Updates SETTINGS upon successful
        authentication.

        Args:
            api_key (str): API key or combined key_id format.
            verbose (bool): Enable verbose logging.
        """
        # 如果 API 密钥包含组合的 key_model 格式，则分割并只保留 API 密钥部分
        # 格式如: "API_KEY_MODEL_ID" -> "API_KEY"
        api_key = api_key.split("_", 1)[0]

        # 设置 API 密钥属性：使用传入的值或从 SETTINGS 中获取
        self.api_key = api_key or SETTINGS.get("api_key", "")

        # 如果提供了 API 密钥
        if self.api_key:
            # 如果提供的 API 密钥与 SETTINGS 中的密钥匹配
            if self.api_key == SETTINGS.get("api_key"):
                # 记录用户已经登录
                if verbose:
                    LOGGER.info(f"{PREFIX}Authenticated ✅")
                return
            else:
                # 尝试使用提供的 API 密钥进行身份验证
                success = self.authenticate()
        # 如果没有提供 API 密钥且当前环境是 Google Colab
        elif IS_COLAB:
            # 尝试使用浏览器 Cookie 进行身份验证
            success = self.auth_with_cookies()
        else:
            # 请求用户输入 API 密钥
            success = self.request_api_key()

        # 如果身份验证成功，更新 SETTINGS 中的 API 密钥
        if success:
            SETTINGS.update({"api_key": self.api_key})
            # 记录新的登录成功
            if verbose:
                LOGGER.info(f"{PREFIX}New authentication successful ✅")
        elif verbose:
            LOGGER.info(f"{PREFIX}Get API key from {API_KEY_URL} and then run 'yolo login API_KEY'")

    def request_api_key(self, max_attempts: int = 3) -> bool:
        """Prompt the user to input their API key.

        Args:
            max_attempts (int): Maximum number of authentication attempts.

        Returns:
            (bool): True if authentication is successful, False otherwise.
        """
        import getpass  # 导入 getpass 模块以安全地获取密码输入

        # 循环尝试多次身份验证
        for attempts in range(max_attempts):
            LOGGER.info(f"{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}")
            # 提示用户输入 API 密钥（输入不会显示在屏幕上）
            input_key = getpass.getpass(f"Enter API key from {API_KEY_URL} ")
            # 移除可能存在的模型 ID 部分，只保留 API 密钥
            self.api_key = input_key.split("_", 1)[0]
            # 尝试进行身份验证
            if self.authenticate():
                return True
        # 如果所有尝试都失败，抛出连接错误
        raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))

    def authenticate(self) -> bool:
        """Attempt to authenticate with the server using either id_token or API key.

        Returns:
            (bool): True if authentication is successful, False otherwise.
        """
        import requests  # 作用域限定的导入，因为 requests 是慢速导入

        try:
            # 获取身份验证头（使用海象运算符同时赋值和判断）
            if header := self.get_auth_header():
                # 向 HUB API 发送身份验证请求
                r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)
                # 检查响应中的成功标志
                if not r.json().get("success", False):
                    raise ConnectionError("Unable to authenticate.")
                return True
            # 如果没有身份验证头，抛出错误
            raise ConnectionError("User has not authenticated locally.")
        except ConnectionError:
            # 重置无效的身份验证信息
            self.id_token = self.api_key = False
            LOGGER.warning(f"{PREFIX}Invalid API key")
            return False

    def auth_with_cookies(self) -> bool:
        """Attempt to fetch authentication via cookies and set id_token.

        User must be logged in to HUB and running in a supported browser.

        Returns:
            (bool): True if authentication is successful, False otherwise.
        """
        if not IS_COLAB:
            # 目前仅支持在 Colab 环境中使用 Cookie 身份验证
            return False
        try:
            # 使用浏览器凭据请求自动身份验证
            authn = request_with_credentials(f"{HUB_API_ROOT}/v1/auth/auto")
            if authn.get("success", False):
                # 从响应中提取 ID 令牌
                self.id_token = authn.get("data", {}).get("idToken", None)
                # 使用 ID 令牌进行身份验证
                self.authenticate()
                return True
            raise ConnectionError("Unable to fetch browser authentication details.")
        except ConnectionError:
            # 重置无效的 ID 令牌
            self.id_token = False
            return False

    def get_auth_header(self):
        """Get the authentication header for making API requests.

        Returns:
            (dict | None): The authentication header if id_token or API key is set, None otherwise.
        """
        if self.id_token:
            # 如果有 ID 令牌，使用 Bearer 认证
            return {"authorization": f"Bearer {self.id_token}"}
        elif self.api_key:
            # 如果有 API 密钥，使用 x-api-key 认证
            return {"x-api-key": self.api_key}
