import io
import os
from typing import Any

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []  # Torch模块__path__._path问题: https://github.com/datalab-to/marker/issues/442


class Inference:
    """
    推理(Inference)类：执行目标检测、图像分类、图像分割和姿态估计推理

    该类提供加载模型、配置设置、上传视频文件以及使用Streamlit和Ultralytics YOLO模型执行实时推理的功能。

    核心功能：
    1. 提供基于Streamlit的Web界面
    2. 支持多种输入源（网络摄像头、视频、图像）
    3. 实时目标检测和追踪
    4. 可配置的检测参数（置信度、IOU阈值等）
    5. 类别过滤和模型选择

    属性:
        st (module): 用于创建UI的Streamlit模块
        temp_dict (dict): 存储模型路径和其他配置的临时字典
        model_path (str): 加载的模型路径
        model (YOLO): YOLO模型实例
        source (str): 选择的视频源（网络摄像头或视频文件）
        enable_trk (bool): 启用追踪选项
        conf (float): 检测的置信度阈值
        iou (float): 非极大值抑制的IoU阈值
        org_frame (Any): 显示原始帧的容器
        ann_frame (Any): 显示标注帧的容器
        vid_file_name (str | int): 上传的视频文件名或网络摄像头索引
        selected_ind (list[int]): 检测的选定类别索引列表

    方法:
        web_ui: 设置带有自定义HTML元素的Streamlit Web界面
        sidebar: 为模型和推理设置配置Streamlit侧边栏
        source_upload: 通过Streamlit界面处理视频文件上传
        configure: 配置模型并加载选定的类别用于推理
        inference: 执行实时目标检测推理

    使用示例:
        使用自定义模型创建Inference实例
        >>> inf = Inference(model="path/to/model.pt")
        >>> inf.inference()

        使用默认设置创建Inference实例
        >>> inf = Inference()
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化Inference类，检查Streamlit要求并设置模型路径

        Args:
            **kwargs (Any): 模型配置的附加关键字参数
        """
        check_requirements("streamlit>=1.29.0")  # 限定导入范围以提高ultralytics包加载速度
        import streamlit as st

        self.st = st  # Streamlit模块的引用
        self.source = None  # 视频源选择（网络摄像头或视频文件）
        self.img_file_names = []  # 图像文件名列表
        self.enable_trk = False  # 切换目标追踪的标志
        self.conf = 0.25  # 检测的置信度阈值
        self.iou = 0.45  # 非极大值抑制的交并比(IoU)阈值
        self.org_frame = None  # 原始帧显示的容器
        self.ann_frame = None  # 标注帧显示的容器
        self.vid_file_name = None  # 视频文件名或网络摄像头索引
        self.selected_ind: list[int] = []  # 检测的选定类别索引列表
        self.model = None  # YOLO模型实例

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # 模型文件路径
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self) -> None:
        """
        设置带有自定义HTML元素的Streamlit Web界面

        该方法配置Streamlit应用的视觉元素：
        1. 隐藏默认主菜单
        2. 设置应用主标题
        3. 设置应用副标题
        4. 配置页面布局为宽屏模式
        """
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # 隐藏主菜单样式

        # Streamlit应用的主标题
        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Streamlit应用的副标题
        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam, videos, and images
        with the power of Ultralytics YOLO! 🚀</h5></div>"""

        # 设置HTML页面配置并添加自定义HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        """
        为模型和推理设置配置Streamlit侧边栏

        侧边栏设置包括：
        1. 显示Ultralytics Logo
        2. 输入源选择（网络摄像头/视频/图像）
        3. 追踪选项（仅视频/摄像头）
        4. 置信度阈值滑块（0.0-1.0）
        5. IoU阈值滑块（0.0-1.0）
        6. 创建原始帧和标注帧的显示容器
        """
        with self.st.sidebar:  # 添加Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # 向垂直设置菜单添加元素
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )  # 添加源选择下拉菜单
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"  # 启用目标追踪
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # 置信度滑块
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # NMS阈值滑块

        if self.source != "image":  # 仅为视频/网络摄像头创建列
            col1, col2 = self.st.columns(2)  # 创建两列用于显示帧
            self.org_frame = col1.empty()  # 原始帧容器
            self.ann_frame = col2.empty()  # 标注帧容器

    def source_upload(self) -> None:
        """
        通过Streamlit界面处理视频文件上传

        处理流程：
        1. 根据选择的源类型执行不同操作：
           - 视频：上传视频文件并保存为临时文件
           - 网络摄像头：使用摄像头索引0
           - 图像：上传多个图像文件并保存为临时文件
        2. 存储文件路径或摄像头索引供后续使用
        """
        from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS  # 限定导入范围

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO对象
                with open("ultralytics.mp4", "wb") as out:  # 以字节模式打开临时文件
                    out.write(g.read())  # 将字节读入文件
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # 使用网络摄像头索引0
        elif self.source == "image":
            import tempfile  # 限定导入范围

            if imgfiles := self.st.sidebar.file_uploader(
                "Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:  # 将每个上传的图像保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        """
        配置模型并加载选定的类别用于推理

        配置流程：
        1. 创建可用模型列表（从GITHUB_ASSETS_STEMS）
        2. 如果提供了自定义模型路径，添加到列表顶部
        3. 使用下拉菜单让用户选择模型
        4. 加载选定的YOLO模型
        5. 提取模型类别名称
        6. 使用多选框让用户选择要检测的类别
        7. 存储选定类别的索引
        """
        # 为模型选择添加下拉菜单
        M_ORD, T_ORD = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"], ["", "-seg", "-pose", "-obb", "-cls"]
        available_models = sorted(
            [
                x.replace("yolo", "YOLO")
                for x in GITHUB_ASSETS_STEMS
                if any(x.startswith(b) for b in M_ORD) and "grayscale" not in x
            ],
            key=lambda x: (M_ORD.index(x[:7].lower()), T_ORD.index(x[7:].lower() or "")),
        )
        if self.model_path:  # 在available_models中插入用户提供的自定义模型
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            if selected_model.endswith((".pt", ".onnx", ".torchscript", ".mlpackage", ".engine")) or any(
                fmt in selected_model for fmt in ("openvino_model", "rknn_model")
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"  # 如果函数调用期间未提供模型，默认为.pt
            self.model = YOLO(model_path)  # 加载YOLO模型
            class_names = list(self.model.names.values())  # 将字典转换为类别名称列表
        self.st.success("Model loaded successfully!")

        # 带有类别名称的多选框并获取选定类别的索引
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # 确保selected_options是列表
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        """
        对上传的图像执行推理

        处理流程：
        1. 遍历所有上传的图像文件
        2. 加载并显示原始图像
        3. 使用YOLO模型执行推理
        4. 显示标注后的图像
        5. 清理临时文件
        """
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)  # 加载并显示原始图像
            if image is not None:
                self.st.markdown(f"#### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Image")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="Predicted Image")
                try:  # 清理临时文件
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass  # 文件不存在，忽略
            else:
                self.st.error("Could not load the uploaded image.")

    def inference(self) -> None:
        """
        对视频或网络摄像头Feed执行实时目标检测推理

        主流程：
        1. 初始化Web界面
        2. 创建侧边栏配置
        3. 处理源上传
        4. 配置模型和类别
        5. 等待用户点击"Start"按钮
        6. 根据源类型执行不同推理：
           - 图像：批量处理所有上传的图像
           - 视频/摄像头：循环读取帧并实时处理
        7. 显示原始帧和标注帧
        8. 支持通过"Stop"按钮停止推理
        """
        self.web_ui()  # 初始化Web界面
        self.sidebar()  # 创建侧边栏
        self.source_upload()  # 上传视频源
        self.configure()  # 配置应用

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file to perform inference.")
                return

            stop_button = self.st.sidebar.button("Stop")  # 停止推理的按钮
            cap = cv2.VideoCapture(self.vid_file_name)  # 捕获视频
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                    break

                # 使用模型处理帧
                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                annotated_frame = results[0].plot()  # 在帧上添加标注

                if stop_button:
                    cap.release()  # 释放捕获
                    self.st.stop()  # 停止streamlit应用

                self.org_frame.image(frame, channels="BGR", caption="Original Frame")  # 显示原始帧
                self.ann_frame.image(annotated_frame, channels="BGR", caption="Predicted Frame")  # 显示处理后的帧

            cap.release()  # 释放捕获
        cv2.destroyAllWindows()  # 销毁所有OpenCV窗口


if __name__ == "__main__":
    import sys  # 导入sys模块以访问命令行参数

    # 检查是否提供了模型名称作为命令行参数
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # 如果提供，将第一个参数分配为模型名称
    # 创建Inference类的实例并运行推理
    Inference(model=model).inference()
