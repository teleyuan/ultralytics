from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免某些系统上的OpenMP冲突


class VisualAISearch:
    """
    视觉AI搜索(VisualAISearch)类：语义图像搜索系统

    该类利用OpenCLIP生成高质量的图像和文本嵌入向量，并使用FAISS进行快速的基于相似度的检索。
    它将图像和文本嵌入对齐到共享的语义空间中，使用户能够使用自然语言查询高精度、高速地搜索大量图像集合。

    核心功能：
    1. 使用CLIP模型提取图像和文本的语义特征
    2. 构建和管理FAISS索引以实现快速相似度搜索
    3. 支持自然语言查询图像
    4. 按相似度排序返回搜索结果

    属性:
        data (str): 包含图像的目录路径
        device (str): 计算设备，例如 'cpu' 或 'cuda'
        faiss_index (str): FAISS索引文件的路径
        data_path_npy (str): 存储图像路径的numpy文件路径
        data_dir (Path): 数据目录的路径对象
        model: 加载的CLIP模型
        index: 用于相似度搜索的FAISS索引
        image_paths (list[str]): 图像文件路径列表

    方法:
        extract_image_feature: 从图像中提取CLIP嵌入向量
        extract_text_feature: 从文本中提取CLIP嵌入向量
        load_or_build_index: 加载现有的FAISS索引或构建新索引
        search: 执行语义搜索以查找相似图像

    使用示例:
        >>> from ultralytics.solutions import VisualAISearch
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("一只坐在椅子上的猫", k=10)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化VisualAISearch类，配置FAISS索引和CLIP模型

        Args:
            **kwargs (Any): 配置参数，包括:
                - data (str): 图像目录路径
                - device (str): 计算设备（'cpu' 或 'cuda'）
        """
        assert TORCH_2_4, f"VisualAISearch需要torch>=2.4（当前torch=={TORCH_VERSION}）"
        from ultralytics.nn.text_model import build_text_model

        check_requirements("faiss-cpu")

        self.faiss = __import__("faiss")
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.data_dir = Path(kwargs.get("data", "images"))
        self.device = select_device(kwargs.get("device", "cpu"))

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"未找到{self.data_dir}。正在从{ASSETS_URL}/images.zip下载images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model = build_text_model("clip:ViT-B/32", device=self.device)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        """
        从给定的图像路径提取CLIP图像嵌入向量

        Args:
            path (Path): 图像文件路径

        Returns:
            (np.ndarray): 图像的CLIP嵌入向量
        """
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """
        从给定的文本查询提取CLIP文本嵌入向量

        Args:
            text (str): 文本查询字符串

        Returns:
            (np.ndarray): 文本的CLIP嵌入向量
        """
        return self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()

    def load_or_build_index(self) -> None:
        """
        加载现有的FAISS索引或从图像特征构建新索引

        该方法检查磁盘上是否存在FAISS索引和图像路径文件。如果找到，直接加载它们。
        否则，通过提取数据目录中所有图像的特征来构建新索引，归一化特征，并保存索引和图像路径以供将来使用。

        处理流程：
        1. 检查是否存在已保存的索引文件
        2. 如果存在，加载索引和图像路径列表
        3. 如果不存在，遍历图像目录：
           - 提取每张图像的CLIP特征向量
           - 收集所有特征向量
        4. 归一化特征向量（用于余弦相似度计算）
        5. 创建FAISS索引并添加向量
        6. 保存索引和路径列表到磁盘
        """
        # 检查FAISS索引和对应的图像路径是否已经存在
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("正在加载现有的FAISS索引...")
            self.index = self.faiss.read_index(self.faiss_index)  # 从磁盘加载FAISS索引
            self.image_paths = np.load(self.data_path_npy)  # 加载保存的图像路径列表
            return  # 索引成功加载，退出函数

        # 如果索引不存在，从头开始构建它
        LOGGER.info("正在从图像构建FAISS索引...")
        vectors = []  # 用于存储图像特征向量的列表

        # 遍历数据目录中的所有图像文件
        for file in self.data_dir.iterdir():
            # 跳过非有效图像格式的文件
            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:
                # 提取图像的特征向量并添加到列表
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)  # 存储对应的图像名称
            except Exception as e:
                LOGGER.warning(f"跳过{file.name}: {e}")

        # 如果没有成功创建任何向量，抛出错误
        if not vectors:
            raise RuntimeError("无法生成图像嵌入向量。")

        vectors = np.vstack(vectors).astype("float32")  # 将所有向量堆叠成NumPy数组并转换为float32
        self.faiss.normalize_L2(vectors)  # 将向量归一化为单位长度，用于余弦相似度计算

        self.index = self.faiss.IndexFlatIP(vectors.shape[1])  # 使用内积创建新的FAISS索引
        self.index.add(vectors)  # 将归一化后的向量添加到FAISS索引
        self.faiss.write_index(self.index, self.faiss_index)  # 将新构建的FAISS索引保存到磁盘
        np.save(self.data_path_npy, np.array(self.image_paths))  # 将图像路径列表保存到磁盘

        LOGGER.info(f"已索引{len(self.image_paths)}张图像。")

    def search(self, query: str, k: int = 30, similarity_thresh: float = 0.1) -> list[str]:
        """
        返回与给定查询语义最相似的前k张图像

        该方法使用自然语言查询在索引的图像集合中搜索语义相似的图像。
        它提取查询文本的CLIP嵌入，在FAISS索引中搜索最相似的向量，并返回按相似度排序的结果。

        Args:
            query (str): 用于搜索的自然语言文本查询
            k (int, optional): 要返回的最大结果数量，默认30
            similarity_thresh (float, optional): 过滤结果的最小相似度阈值，默认0.1

        Returns:
            (list[str]): 按相似度分数排序的图像文件名列表

        使用示例:
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("红色汽车", k=5, similarity_thresh=0.2)
            >>> print(results)
        """
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)

        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(index[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\n排序结果:")
        for name, score in results:
            LOGGER.info(f"  - {name} | 相似度: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        """
        搜索函数的直接调用接口

        Args:
            query (str): 搜索查询文本

        Returns:
            (list[str]): 搜索结果列表
        """
        return self.search(query)


class SearchApp:
    """
    搜索应用(SearchApp)类：基于Flask的语义图像搜索Web界面

    该类提供了一个清晰、响应式的前端界面，使用户能够输入自然语言查询并即时查看从索引数据库中检索到的最相关图像。

    核心功能：
    1. 提供Web界面用于图像搜索
    2. 集成VisualAISearch后端进行语义搜索
    3. 展示搜索结果和相似度分数
    4. 支持本地图像服务

    属性:
        render_template: Flask模板渲染函数
        request: Flask请求对象
        searcher (VisualAISearch): VisualAISearch类的实例
        app (Flask): Flask应用程序实例

    方法:
        index: 处理用户查询并显示搜索结果
        run: 启动Flask Web应用程序

    使用示例:
        >>> from ultralytics.solutions import SearchApp
        >>> app = SearchApp(data="path/to/images", device="cuda")
        >>> app.run(debug=True)
    """

    def __init__(self, data: str = "images", device: str | None = None) -> None:
        """
        使用VisualAISearch后端初始化SearchApp

        Args:
            data (str, optional): 包含要索引和搜索的图像的目录路径，默认"images"
            device (str, optional): 运行推理的设备（例如'cpu'、'cuda'）
        """
        check_requirements("flask>=3.0.1")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),  # 用于提供图像的绝对路径
            static_url_path="/images",  # 图像的URL前缀
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self) -> str:
        """
        处理用户查询并在Web界面中显示搜索结果

        Returns:
            (str): 渲染后的HTML模板
        """
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug: bool = False) -> None:
        """
        启动Flask Web应用程序服务器

        Args:
            debug (bool): 是否以调试模式运行，默认False
        """
        self.app.run(debug=debug)
