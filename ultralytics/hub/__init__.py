# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics HUB 集成模块

该模块提供了与 Ultralytics HUB 云平台交互的核心功能，支持模型训练、推理和管理的云服务集成。
Ultralytics HUB 是一个用于训练、部署和管理 YOLO 模型的无代码平台。

主要功能:
    - 用户身份验证（登录/登出）
    - HUB 训练会话管理
    - 模型导出和获取
    - 数据集验证和上传
    - 模型重置功能

导出的类:
    HUBTrainingSession: HUB 训练会话管理类

导出的函数:
    login: 登录到 Ultralytics HUB
    logout: 从 Ultralytics HUB 登出
    reset_model: 重置已训练的模型
    export_model: 导出模型到指定格式
    get_export: 获取导出的模型
    export_fmts_hub: 获取 HUB 支持的导出格式列表
    check_dataset: 检查数据集是否符合 HUB 上传要求

相关链接:
    - HUB 主页: https://hub.ultralytics.com
    - HUB 文档: https://docs.ultralytics.com/hub/
"""

from __future__ import annotations  # 支持类型注解中的前向引用

# 导入数据集统计工具
from ultralytics.data.utils import HUBDatasetStats  # HUB 数据集统计和验证

# 导入 HUB 核心组件
from ultralytics.hub.auth import Auth  # 身份验证类
from ultralytics.hub.session import HUBTrainingSession  # 训练会话管理类
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX  # HUB 相关常量和工具

# 导入通用工具
from ultralytics.utils import LOGGER, SETTINGS, checks  # 日志、设置和检查工具

# 定义模块的公共接口
__all__ = (
    "HUB_WEB_ROOT",  # HUB 网页根 URL
    "PREFIX",  # 日志消息前缀
    "HUBTrainingSession",  # HUB 训练会话类
    "check_dataset",  # 数据集检查函数
    "export_fmts_hub",  # HUB 支持的导出格式
    "export_model",  # 模型导出函数
    "get_export",  # 获取导出模型函数
    "login",  # 登录函数
    "logout",  # 登出函数
    "reset_model",  # 模型重置函数
)


def login(api_key: str | None = None, save: bool = True) -> bool:
    """Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY
    environment variable if successfully authenticated.

    Args:
        api_key (str, optional): API key to use for authentication. If not provided, it will be retrieved from SETTINGS
            or HUB_API_KEY environment variable.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    # 检查并安装必需的 hub-sdk 包（版本 >= 0.0.12）
    checks.check_requirements("hub-sdk>=0.0.12")
    from hub_sdk import HUBClient  # 导入 HUB 客户端

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # 设置 API 密钥获取的重定向 URL
    saved_key = SETTINGS.get("api_key")  # 从设置中获取已保存的 API 密钥
    active_key = api_key or saved_key  # 使用提供的密钥或已保存的密钥
    # 设置认证凭据：如果有有效的密钥则使用，否则设为 None
    credentials = {"api_key": active_key} if active_key and active_key != "" else None

    client = HUBClient(credentials)  # 初始化 HUB 客户端

    if client.authenticated:
        # 成功通过 HUB 身份验证

        if save and client.api_key != saved_key:
            # 如果需要保存且密钥与已保存的不同，则更新设置
            SETTINGS.update({"api_key": client.api_key})

        # 根据是否提供了新密钥设置日志消息
        log_message = (
            "New authentication successful ✅" if client.api_key == api_key or not credentials else "Authenticated ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")

        return True
    else:
        # HUB 身份验证失败
        LOGGER.info(f"{PREFIX}Get API key from {api_key_url} and then run 'yolo login API_KEY'")
        return False


def logout():
    """Log out of Ultralytics HUB by removing the API key from the settings file."""
    SETTINGS["api_key"] = ""  # 清空设置中保存的 API 密钥
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo login'.")


def reset_model(model_id: str = ""):
    """Reset a trained model to an untrained state."""
    import requests  # 作用域限定的导入，因为 requests 是慢速导入

    # 发送模型重置请求到 HUB API
    r = requests.post(f"{HUB_API_ROOT}/model-reset", json={"modelId": model_id}, headers={"x-api-key": Auth().api_key})
    if r.status_code == 200:
        # 模型重置成功
        LOGGER.info(f"{PREFIX}Model reset successfully")
        return
    # 模型重置失败
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")


def export_fmts_hub():
    """Return a list of HUB-supported export formats."""
    from ultralytics.engine.exporter import export_formats  # 导入导出格式函数

    # 返回 HUB 支持的导出格式列表（包括标准格式和 Ultralytics 专用格式）
    return [*list(export_formats()["Argument"][1:]), "ultralytics_tflite", "ultralytics_coreml"]


def export_model(model_id: str = "", format: str = "torchscript"):
    """Export a model to a specified format for deployment via the Ultralytics HUB API.

    Args:
        model_id (str): The ID of the model to export. An empty string will use the default model.
        format (str): The format to export the model to. Must be one of the supported formats returned by
            export_fmts_hub().

    Raises:
        AssertionError: If the specified format is not supported or if the export request fails.

    Examples:
        >>> from ultralytics import hub
        >>> hub.export_model(model_id="your_model_id", format="torchscript")
    """
    import requests  # 作用域限定的导入，因为 requests 是慢速导入

    # 验证导出格式是否在支持的格式列表中
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    # 发送模型导出请求到 HUB API
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    # 验证请求是否成功
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id: str = "", format: str = "torchscript"):
    """Retrieve an exported model in the specified format from Ultralytics HUB using the model ID.

    Args:
        model_id (str): The ID of the model to retrieve from Ultralytics HUB.
        format (str): The export format to retrieve. Must be one of the supported formats returned by export_fmts_hub().

    Returns:
        (dict): JSON response containing the exported model information.

    Raises:
        AssertionError: If the specified format is not supported or if the API request fails.

    Examples:
        >>> from ultralytics import hub
        >>> result = hub.get_export(model_id="your_model_id", format="torchscript")
    """
    import requests  # 作用域限定的导入，因为 requests 是慢速导入

    # 验证导出格式是否在支持的格式列表中
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    # 发送获取导出模型的请求到 HUB API
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    # 验证请求是否成功
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()  # 返回 JSON 格式的响应


def check_dataset(path: str, task: str) -> None:
    """Check HUB dataset Zip file for errors before upload.

    Args:
        path (str): Path to data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify', 'obb'.

    Examples:
        >>> from ultralytics.hub import check_dataset
        >>> check_dataset("path/to/coco8.zip", task="detect")  # detect dataset
        >>> check_dataset("path/to/coco8-seg.zip", task="segment")  # segment dataset
        >>> check_dataset("path/to/coco8-pose.zip", task="pose")  # pose dataset
        >>> check_dataset("path/to/dota8.zip", task="obb")  # OBB dataset
        >>> check_dataset("path/to/imagenet10.zip", task="classify")  # classification dataset

    Notes:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
        i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
    """
    # 使用 HUBDatasetStats 验证数据集并生成统计信息的 JSON
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.")
