"""
文本编码模型模块

这个模块提供了用于视觉-语言任务的文本编码模型实现。
主要用于将文本描述编码为特征向量，以便与图像特征进行对齐和匹配。

主要模型:
    - CLIP: OpenAI 的对比语言-图像预训练模型
    - MobileCLIP: Apple 的轻量级 CLIP 模型
    - MobileCLIPTS: MobileCLIP 的 TorchScript 版本

应用场景:
    - 开放词汇目标检测（YOLO-World）
    - 文本-图像检索
    - 零样本分类
    - 视觉提示学习（YOLOE）
"""

from __future__ import annotations  # 启用延迟类型注解评估，支持前向引用

# 导入标准库
from abc import abstractmethod  # 用于定义抽象基类
from pathlib import Path  # 用于跨平台的路径操作

# 导入 PyTorch 核心库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块
from PIL import Image  # Python 图像处理库

# 导入 Ultralytics 工具函数
from ultralytics.utils import checks  # 依赖检查工具
from ultralytics.utils.torch_utils import smart_inference_mode  # 智能推理模式装饰器

# 尝试导入 CLIP 库，如果未安装则自动安装
try:
    import clip  # OpenAI CLIP 库
except ImportError:
    # 自动安装 Ultralytics 维护的 CLIP 版本
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    """文本编码模型的抽象基类

    该类定义了用于视觉-语言任务的文本编码模型接口。子类必须实现 tokenize 和 encode_text 方法以提供文本分词和编码功能。

    方法:
        tokenize: 将输入文本转换为模型处理的标记
        encode_text: 将分词后的文本编码为归一化的特征向量
    """

    def __init__(self):
        """初始化 TextModel 基类"""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """将输入文本转换为模型处理的标记"""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """将分词后的文本编码为归一化的特征向量"""
        pass


class CLIP(TextModel):
    """实现 OpenAI 的 CLIP (对比语言-图像预训练) 文本编码器

    该类提供基于 OpenAI CLIP 模型的文本编码器,可以将文本转换为特征向量,这些向量在共享嵌入空间中与相应的图像特征对齐。

    属性:
        model (clip.model.CLIP): 加载的 CLIP 模型
        device (torch.device): 模型加载的设备

    方法:
        tokenize: 将输入文本转换为 CLIP 标记
        encode_text: 将分词后的文本编码为归一化的特征向量

    示例:
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 CLIP 文本编码器

        该类使用 OpenAI 的 CLIP 模型实现 TextModel 接口以进行文本编码。它加载指定大小的预训练 CLIP 模型并为文本编码任务做准备。

        参数:
            size (str): 模型大小标识符 (例如 'ViT-B/32')
            device (torch.device): 加载模型的设备
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 CLIP 标记

        参数:
            texts (str | list[str]): 要分词的输入文本或文本列表
            truncate (bool, optional): 是否截断超过 CLIP 上下文长度的文本。默认为 True 以避免过长输入导致的 RuntimeError,
                同时仍然允许显式选择不截断

        返回:
            (torch.Tensor): 形状为 (batch_size, context_length) 的分词文本张量,可用于模型处理

        示例:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
            >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # 强制执行严格的长度检查
            >>> print(strict_tokens.shape)  # 与 tokens 形状/内容相同,因为提示少于 77 个标记
        """
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量

        该方法通过 CLIP 模型处理分词的文本输入以生成特征向量,然后将其归一化为单位长度。这些归一化向量可用于文本-图像相似度比较。

        参数:
            texts (torch.Tensor): 分词的文本输入,通常使用 tokenize() 方法创建
            dtype (torch.dtype, optional): 输出特征的数据类型

        返回:
            (torch.Tensor): 单位长度的归一化文本特征向量 (L2 范数 = 1)

        示例:
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将预处理的图像编码为归一化的特征向量

        该方法通过 CLIP 模型处理预处理的图像输入以生成特征向量,然后将其归一化为单位长度。这些归一化向量可用于文本-图像相似度比较。

        参数:
            image (PIL.Image | torch.Tensor): 预处理的图像输入。如果提供 PIL Image,它将使用模型的图像预处理函数转换为张量
            dtype (torch.dtype, optional): 输出特征的数据类型

        返回:
            (torch.Tensor): 单位长度的归一化图像特征向量 (L2 范数 = 1)

        示例:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """实现 Apple 的 MobileCLIP 文本编码器以进行高效文本编码

    该类使用 Apple 的 MobileCLIP 模型实现 TextModel 接口,为视觉-语言任务提供高效的文本编码能力,与标准 CLIP 模型相比具有更低的计算要求。

    属性:
        model (mobileclip.model.MobileCLIP): 加载的 MobileCLIP 模型
        tokenizer (callable): 用于处理文本输入的分词器函数
        device (torch.device): 模型加载的设备
        config_size_map (dict): 从大小标识符到模型配置名称的映射

    方法:
        tokenize: 将输入文本转换为 MobileCLIP 标记
        encode_text: 将分词后的文本编码为归一化的特征向量

    示例:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 MobileCLIP 文本编码器

        该类使用 Apple 的 MobileCLIP 模型实现 TextModel 接口以进行高效文本编码。

        参数:
            size (str): 模型大小标识符 (例如 's0', 's1', 's2', 'b', 'blt')
            device (torch.device): 加载模型的设备
        """
        try:
            import mobileclip
        except ImportError:
            # 优先使用 Ultralytics fork,因为 Apple MobileCLIP 仓库中的 torchvision 版本不正确
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP 标记

        参数:
            texts (list[str]): 要分词的文本字符串列表

        返回:
            (torch.Tensor): 形状为 (batch_size, sequence_length) 的分词文本输入

        示例:
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量

        参数:
            texts (torch.Tensor): 分词的文本输入
            dtype (torch.dtype, optional): 输出特征的数据类型

        返回:
            (torch.Tensor): 应用 L2 归一化的归一化文本特征向量

        示例:
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型大小
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """加载 MobileCLIP 的 TorchScript 追踪版本

    该类使用 TorchScript 格式的 Apple MobileCLIP 模型实现 TextModel 接口,为视觉-语言任务提供高效的文本编码能力,具有优化的推理性能。

    属性:
        encoder (torch.jit.ScriptModule): 加载的 TorchScript MobileCLIP 文本编码器
        tokenizer (callable): 用于处理文本输入的分词器函数
        device (torch.device): 模型加载的设备

    方法:
        tokenize: 将输入文本转换为 MobileCLIP 标记
        encode_text: 将分词后的文本编码为归一化的特征向量

    示例:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, device: torch.device):
        """初始化 MobileCLIP TorchScript 文本编码器

        该类使用 TorchScript 格式的 Apple MobileCLIP 模型实现 TextModel 接口,以优化的推理性能进行高效文本编码。

        参数:
            device (torch.device): 加载模型的设备
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset("mobileclip_blt.ts"), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP 标记

        参数:
            texts (list[str]): 要分词的文本字符串列表
            truncate (bool, optional): 是否截断超过分词器上下文长度的文本。默认为 True,与 CLIP 的行为匹配以防止长标题导致运行时失败

        返回:
            (torch.Tensor): 形状为 (batch_size, sequence_length) 的分词文本输入

        示例:
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # 如果超过 77 个标记则抛出 RuntimeError
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将分词后的文本编码为归一化的特征向量

        参数:
            texts (torch.Tensor): 分词的文本输入
            dtype (torch.dtype, optional): 输出特征的数据类型

        返回:
            (torch.Tensor): 应用 L2 归一化的归一化文本特征向量

        示例:
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型大小
        """
        # 注意: 这里不需要进行归一化,因为它已嵌入在 TorchScript 模型中
        return self.encoder(texts).to(dtype)


def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """根据指定的变体构建文本编码模型

    参数:
        variant (str): 格式为 "base:size" 的模型变体 (例如 "clip:ViT-B/32" 或 "mobileclip:s0")
        device (torch.device, optional): 加载模型的设备

    返回:
        (TextModel): 实例化的文本编码模型

    示例:
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    else:
        raise ValueError(f"无法识别的基础模型: '{base}'。支持的基础模型: 'clip', 'mobileclip'。")
