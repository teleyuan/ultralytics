"""
自动后端选择模块

这个模块提供了 AutoBackend 类，用于动态选择和加载不同格式的 YOLO 模型。
它支持多种推理后端和模型格式，使得模型可以在不同平台和硬件上运行。

支持的模型格式:
    - PyTorch (.pt)
    - TorchScript (.torchscript)
    - ONNX Runtime (.onnx)
    - OpenVINO (*_openvino_model/)
    - TensorRT (.engine)
    - CoreML (.mlpackage)
    - TensorFlow SavedModel (*_saved_model/)
    - TensorFlow Lite (.tflite)
    - PaddlePaddle (*_paddle_model/)
    - MNN (.mnn)
    - NCNN (*_ncnn_model/)
    - RKNN (*_rknn_model/)
    - Triton Inference Server (triton://)
    - ExecuTorch (.pte)
    - Axelera (*_axelera_model/)

主要功能:
    - 自动检测模型格式
    - 加载相应的推理引擎
    - 统一的前向传播接口
    - 设备自适应（CPU/GPU）
    - 动态批处理支持
"""

from __future__ import annotations  # 启用延迟类型注解评估

# 导入标准库
import ast  # 抽象语法树，用于安全解析字符串
import json  # JSON 数据处理
import platform  # 平台信息获取
import zipfile  # ZIP 文件处理
from collections import OrderedDict, namedtuple  # 有序字典和命名元组
from pathlib import Path  # 跨平台路径操作
from typing import Any  # 类型提示

# 导入第三方库
import cv2  # OpenCV 图像处理库
import numpy as np  # NumPy 数值计算库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块
from PIL import Image  # Python 图像处理库

# 导入 Ultralytics 工具函数
from ultralytics.utils import ARM64, IS_JETSON, LINUX, LOGGER, PYTHON_VERSION, ROOT, YAML, is_jetson
from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml, is_rockchip
from ultralytics.utils.downloads import attempt_download_asset, is_url  # 资源下载工具
from ultralytics.utils.nms import non_max_suppression  # 非极大值抑制


def check_class_names(names: list | dict) -> dict[int, str]:
    """
    检查并转换类别名称格式

    将类别名称转换为标准的字典格式，键为整数索引，值为字符串名称。
    同时验证类别索引的有效性，并处理 ImageNet 特殊格式（n开头的代码）。

    Args:
        names (list | dict): 类别名称，可以是列表或字典格式

    Returns:
        (dict): 标准格式的类别名称字典，键为整数，值为字符串

    Raises:
        KeyError: 如果类别索引对于数据集大小无效

    Examples:
        >>> check_class_names(['person', 'car', 'bicycle'])
        {0: 'person', 1: 'car', 2: 'bicycle'}
        >>> check_class_names({0: 'person', 1: 'car'})
        {0: 'person', 1: 'car'}
    """
    # 如果是列表格式，转换为字典（索引作为键）
    if isinstance(names, list):
        names = dict(enumerate(names))  # [name1, name2] -> {0: name1, 1: name2}

    if isinstance(names, dict):
        # 确保键为整数，值为字符串
        # 例如: {'0': 'person'} -> {0: 'person'}, {0: True} -> {0: 'True'}
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)

        # 验证类别索引范围：必须是 0 到 n-1
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n} 类数据集需要类别索引 0-{n - 1}，但在数据集 YAML 中定义了无效的类别索引 "
                f"{min(names.keys())}-{max(names.keys())}"
            )

        # 处理 ImageNet 特殊格式：将 'n01440764' 这样的代码转换为可读名称
        if isinstance(names[0], str) and names[0].startswith("n0"):
            # 加载 ImageNet 映射文件，将代码转换为人类可读的名称
            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}

    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """对输入的 YAML 文件应用默认类别名称或返回数字类别名称

    参数:
        data (str | Path, optional): 包含类别名称的 YAML 文件路径

    返回:
        (dict): 将类别索引映射到类别名称的字典
    """
    if data:
        try:
            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # 如果上述操作出错则返回默认值


class AutoBackend(nn.Module):
    """处理使用 Ultralytics YOLO 模型运行推理时的动态后端选择

    AutoBackend 类旨在为各种推理引擎提供抽象层。它支持多种格式,每种格式都有特定的命名约定,如下所示:

        支持的格式和命名约定:
            | 格式                  | 文件后缀          |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |
            | Triton Inference      | triton://model    |
            | ExecuTorch            | *.pte             |
            | Axelera               | *_axelera_model/  |

    属性:
        model (torch.nn.Module): 加载的 YOLO 模型
        device (torch.device): 模型加载的设备 (CPU 或 GPU)
        task (str): 模型执行的任务类型 (detect、segment、classify、pose)
        names (dict): 模型可以检测的类别名称字典
        stride (int): 模型步长,对于 YOLO 模型通常为 32
        fp16 (bool): 模型是否使用半精度 (FP16) 推理
        nhwc (bool): 模型是否期望 NHWC 输入格式而不是 NCHW
        pt (bool): 模型是否为 PyTorch 模型
        jit (bool): 模型是否为 TorchScript 模型
        onnx (bool): 模型是否为 ONNX 模型
        xml (bool): 模型是否为 OpenVINO 模型
        engine (bool): 模型是否为 TensorRT 引擎
        coreml (bool): 模型是否为 CoreML 模型
        saved_model (bool): 模型是否为 TensorFlow SavedModel
        pb (bool): 模型是否为 TensorFlow GraphDef
        tflite (bool): 模型是否为 TensorFlow Lite 模型
        edgetpu (bool): 模型是否为 TensorFlow Edge TPU 模型
        tfjs (bool): 模型是否为 TensorFlow.js 模型
        paddle (bool): 模型是否为 PaddlePaddle 模型
        mnn (bool): 模型是否为 MNN 模型
        ncnn (bool): 模型是否为 NCNN 模型
        imx (bool): 模型是否为 IMX 模型
        rknn (bool): 模型是否为 RKNN 模型
        triton (bool): 模型是否为 Triton Inference Server 模型
        pte (bool): 模型是否为 PyTorch ExecuTorch 模型
        axelera (bool): 模型是否为 Axelera 模型

    方法:
        forward: 对输入图像运行推理
        from_numpy: 将 NumPy 数组转换为模型设备上的张量
        warmup: 使用虚拟输入预热模型
        _model_type: 从文件路径确定模型类型

    示例:
        >>> model = AutoBackend(model="yolo11n.pt", device="cuda")
        >>> results = model(img)
    """

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo11n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化用于推理的 AutoBackend

        参数:
            model (str | torch.nn.Module): 模型权重文件的路径或模块实例
            device (torch.device): 运行模型的设备
            dnn (bool): 对 ONNX 推理使用 OpenCV DNN 模块
            data (str | Path, optional): 包含类别名称的额外 data.yaml 文件的路径
            fp16 (bool): 启用半精度推理。仅在特定后端上支持
            fuse (bool): 融合 Conv2D + BatchNorm 层以进行优化
            verbose (bool): 启用详细日志记录
        """
        super().__init__()
        nn_module = isinstance(model, torch.nn.Module)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            mnn,
            ncnn,
            imx,
            rknn,
            pte,
            axelera,
            triton,
        ) = self._model_type("" if nn_module else model)
        fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu or rknn  # BHWC 格式 (相对于 torch BCHW)
        stride, ch = 32, 3  # 默认步长和通道数
        end2end, dynamic = False, False
        metadata, task = None, None

        # 设置设备
        cuda = isinstance(device, torch.device) and torch.cuda.is_available() and device.type != "cpu"  # 使用 CUDA
        if cuda and not any([nn_module, pt, jit, engine, onnx, paddle]):  # GPU dataloader 格式
            device = torch.device("cpu")
            cuda = False

        # 如果不是本地文件则下载
        w = attempt_download_asset(model) if pt else model  # 权重路径

        # PyTorch (内存或文件)
        if nn_module or pt:
            if nn_module:
                pt = True
                if fuse:
                    if IS_JETSON and is_jetson(jetpack=5):
                        # Jetson Jetpack5 需要在 fuse 之前设置设备 https://github.com/ultralytics/ultralytics/pull/21028
                        model = model.to(device)
                    model = model.fuse(verbose=verbose)
                model = model.to(device)
            else:  # pt 文件
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=device, fuse=fuse)  # 加载模型和检查点

            # 通用 PyTorch 模型处理
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # 仅 pose
            stride = max(int(model.stride.max()), 32)  # 模型步长
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类别名称
            model.half() if fp16 else model.float()
            ch = model.yaml.get("channels", 3)
            for p in model.parameters():
                p.requires_grad = False
            self.model = model  # 显式分配以支持 to()、cpu()、cuda()、half()

        # TorchScript
        elif jit:
            import torchvision  # noqa - https://github.com/ultralytics/ultralytics/pull/19747

            LOGGER.info(f"为 TorchScript 推理加载 {w}...")
            extra_files = {"config.txt": ""}  # 模型元数据
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # 加载元数据字典
                metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

        # ONNX OpenCV DNN
        elif dnn:
            LOGGER.info(f"为 ONNX OpenCV DNN 推理加载 {w}...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)

        # ONNX Runtime 和 IMX
        elif onnx or imx:
            LOGGER.info(f"为 ONNX Runtime 推理加载 {w}...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            # 选择执行提供者: CUDA > CoreML (mps) > CPU
            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": device.index}), "CPUExecutionProvider"]
            elif device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("请求了 CUDA 但 CUDAExecutionProvider 不可用。使用 CPU...")
                    device, cuda = torch.device("cpu"), False
            LOGGER.info(
                f"使用 ONNX Runtime {onnxruntime.__version__} 和 {providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )
            if onnx:
                session = onnxruntime.InferenceSession(w, providers=providers)
            else:
                check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
                w = next(Path(w).glob("*.onnx"))
                LOGGER.info(f"为 ONNX IMX 推理加载 {w}...")
                import mct_quantizers as mctq
                from edgemdt_cl.pytorch.nms import nms_ort  # noqa - 注册自定义 NMS 操作

                session_options = mctq.get_ort_session_options()
                session_options.enable_mem_reuse = False  # 修复 onnxruntime 的形状不匹配
                session = onnxruntime.InferenceSession(w, session_options, providers=["CPUExecutionProvider"])

            output_names = [x.name for x in session.get_outputs()]
            metadata = session.get_modelmeta().custom_metadata_map
            dynamic = isinstance(session.get_outputs()[0].shape[0], str)
            fp16 = "float16" in session.get_inputs()[0].type

            # 为优化推理设置 IO 绑定 (仅 CUDA,CoreML 不支持)
            use_io_binding = not dynamic and cuda
            if use_io_binding:
                io = session.io_binding()
                bindings = []
                for output in session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(device)
                    io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    bindings.append(y_tensor)

        # OpenVINO
        elif xml:
            LOGGER.info(f"为 OpenVINO 推理加载 {w}...")
            check_requirements("openvino>=2024.0.0")
            import openvino as ov

            core = ov.Core()
            device_name = "AUTO"
            if isinstance(device, str) and device.startswith("intel"):
                device_name = device.split(":")[1].upper()  # Intel OpenVINO 设备
                device = torch.device("cpu")
                if device_name not in core.available_devices:
                    LOGGER.warning(f"OpenVINO 设备 '{device_name}' 不可用。使用 'AUTO' 代替。")
                    device_name = "AUTO"
            w = Path(w)
            if not w.is_file():  # 如果不是 *.xml
                w = next(w.glob("*.xml"))  # 从 *_openvino_model 目录获取 *.xml 文件
            ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

            metadata = w.parent / "metadata.yaml"
            if metadata.exists():
                metadata = YAML.load(metadata)
                batch = metadata["batch"]
                dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            # OpenVINO 推理模式有 'LATENCY'、'THROUGHPUT' (不推荐) 或 'CUMULATIVE_THROUGHPUT'
            inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 and dynamic else "LATENCY"
            ov_compiled_model = core.compile_model(
                ov_model,
                device_name=device_name,
                config={"PERFORMANCE_HINT": inference_mode},
            )
            LOGGER.info(
                f"在 {', '.join(ov_compiled_model.get_property('EXECUTION_DEVICES'))} 上使用 OpenVINO {inference_mode} 模式进行 batch={batch} 推理..."
            )
            input_name = ov_compiled_model.input().get_any_name()

        # TensorRT
        elif engine:
            LOGGER.info(f"为 TensorRT 推理加载 {w}...")

            if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
                # 修复错误: 对于 JetPack 4 和 Python <= 3.8.10 的 JetPack 5,`np.bool` 是内置 `bool` 的已弃用别名
                check_requirements("numpy==1.23.5")

            try:  # https://developer.nvidia.com/nvidia-tensorrt-download
                import tensorrt as trt
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # 读取文件
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # 读取元数据长度
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))  # 读取元数据
                    dla = metadata.get("dla", None)
                    if dla is not None:
                        runtime.DLA_core = int(dla)
                except UnicodeDecodeError:
                    f.seek(0)  # 引擎文件可能缺少嵌入的 Ultralytics 元数据
                model = runtime.deserialize_cuda_engine(f.read())  # 读取引擎

            # 模型上下文
            try:
                context = model.create_execution_context()
            except Exception as e:  # model 为 None
                LOGGER.error(f"TensorRT 模型导出的版本与 {trt.__version__} 不同\n")
                raise e

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # 下面会更新默认值
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                # 使用 TRT10+ 或旧版 API 获取张量信息
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    shape = tuple(model.get_tensor_shape(name))
                    profile_shape = tuple(model.get_tensor_profile_shape(name, 0)[2]) if is_input else None
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    shape = tuple(model.get_binding_shape(i))
                    profile_shape = tuple(model.get_profile_shape(0, i)[1]) if is_input else None

                # 处理输入/输出张量
                if is_input:
                    if -1 in shape:
                        dynamic = True
                        if is_trt10:
                            context.set_input_shape(name, profile_shape)
                        else:
                            context.set_binding_shape(i, profile_shape)
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_tensor_shape(name)) if is_trt10 else tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

        # CoreML
        elif coreml:
            check_requirements(
                ["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"]
            )  # 最新的 numpy 2.4.0rc1 破坏了 coremltools 导出
            LOGGER.info(f"为 CoreML 推理加载 {w}...")
            import coremltools as ct

            model = ct.models.MLModel(w)
            dynamic = model.get_spec().description.input[0].type.HasField("multiArrayType")
            metadata = dict(model.user_defined_metadata)

        # TF SavedModel
        elif saved_model:
            LOGGER.info(f"为 TensorFlow SavedModel 推理加载 {w}...")
            import tensorflow as tf

            model = tf.saved_model.load(w)
            metadata = Path(w) / "metadata.yaml"

        # TF GraphDef
        elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"为 TensorFlow GraphDef 推理加载 {w}...")
            import tensorflow as tf

            from ultralytics.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """包装冻结图以用于部署"""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # 包装
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
            try:  # 在 GraphDef 旁边的 SavedModel 中查找元数据
                metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))
            except StopIteration:
                pass

        # TFLite 或 TFLite Edge TPU
        elif tflite or edgetpu:  # https://ai.google.dev/edge/litert/microcontrollers/python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                device = device[3:] if str(device).startswith("tpu") else ":0"
                LOGGER.info(f"在设备 {device[1:]} 上为 TensorFlow Lite Edge TPU 推理加载 {w}...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(
                    model_path=w,
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                device = "cpu"  # 必需,否则 PyTorch 会尝试使用错误的设备
            else:  # TFLite
                LOGGER.info(f"为 TensorFlow Lite 推理加载 {w}...")
                interpreter = Interpreter(model_path=w)  # 加载 TFLite 模型
            interpreter.allocate_tensors()  # 分配张量
            input_details = interpreter.get_input_details()  # 输入
            output_details = interpreter.get_output_details()  # 输出
            # 加载元数据
            try:
                with zipfile.ZipFile(w, "r") as zf:
                    name = zf.namelist()[0]
                    contents = zf.read(name).decode("utf-8")
                    if name == "metadata.json":  # Python>=3.12 的自定义 Ultralytics 元数据字典
                        metadata = json.loads(contents)
                    else:
                        metadata = ast.literal_eval(contents)  # Python<=3.11 的默认 tflite-support 元数据
            except (zipfile.BadZipFile, SyntaxError, ValueError, json.JSONDecodeError):
                pass

        # TF.js
        elif tfjs:
            raise NotImplementedError("当前不支持 Ultralytics TF.js 推理。")

        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"为 PaddlePaddle 推理加载 {w}...")
            check_requirements(
                "paddlepaddle-gpu"
                if torch.cuda.is_available()
                else "paddlepaddle==3.0.0"  # 为 ARM64 固定 3.0.0
                if ARM64
                else "paddlepaddle>=3.0.0"
            )
            import paddle.inference as pdi

            w = Path(w)
            model_file, params_file = None, None
            if w.is_dir():
                model_file = next(w.rglob("*.json"), None)
                params_file = next(w.rglob("*.pdiparams"), None)
            elif w.suffix == ".pdiparams":
                model_file = w.with_name("model.json")
                params_file = w

            if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
                raise FileNotFoundError(f"在 {w} 中未找到 Paddle 模型。需要 .json 和 .pdiparams 两个文件。")

            config = pdi.Config(str(model_file), str(params_file))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
            metadata = w / "metadata.yaml"

        # MNN
        elif mnn:
            LOGGER.info(f"为 MNN 推理加载 {w}...")
            check_requirements("MNN")  # 需要 MNN
            import os

            import MNN

            config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
            rt = MNN.nn.create_runtime_manager((config,))
            net = MNN.nn.load_module_from_file(w, [], [], runtime_manager=rt, rearrange=True)

            def torch_to_mnn(x):
                return MNN.expr.const(x.data_ptr(), x.shape)

            metadata = json.loads(net.get_info()["bizCode"])

        # NCNN
        elif ncnn:
            LOGGER.info(f"为 NCNN 推理加载 {w}...")
            check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn", cmds="--no-deps")
            import ncnn as pyncnn

            net = pyncnn.Net()
            net.opt.use_vulkan_compute = cuda
            w = Path(w)
            if not w.is_file():  # 如果不是 *.param
                w = next(w.glob("*.param"))  # 从 *_ncnn_model 目录获取 *.param 文件
            net.load_param(str(w))
            net.load_model(str(w.with_suffix(".bin")))
            metadata = w.parent / "metadata.yaml"

        # NVIDIA Triton Inference Server
        elif triton:
            check_requirements("tritonclient[all]")
            from ultralytics.utils.triton import TritonRemoteModel

            model = TritonRemoteModel(w)
            metadata = model.metadata

        # RKNN
        elif rknn:
            if not is_rockchip():
                raise OSError("RKNN 推理仅在 Rockchip 设备上支持。")
            LOGGER.info(f"为 RKNN 推理加载 {w}...")
            check_requirements("rknn-toolkit-lite2")
            from rknnlite.api import RKNNLite

            w = Path(w)
            if not w.is_file():  # 如果不是 *.rknn
                w = next(w.rglob("*.rknn"))  # 从 *_rknn_model 目录获取 *.rknn 文件
            rknn_model = RKNNLite()
            rknn_model.load_rknn(str(w))
            rknn_model.init_runtime()
            metadata = w.parent / "metadata.yaml"

        # Axelera
        elif axelera:
            import os

            if not os.environ.get("AXELERA_RUNTIME_DIR"):
                LOGGER.warning(
                    "Axelera 运行时环境未激活。"
                    "\n请运行: source /opt/axelera/sdk/latest/axelera_activate.sh"
                    "\n\n如果失败,请验证驱动安装: https://docs.ultralytics.com/integrations/axelera/#axelera-driver-installation"
                )
            try:
                from axelera.runtime import op
            except ImportError:
                check_requirements(
                    "axelera_runtime2==0.1.2",
                    cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi",
                )
            from axelera.runtime import op

            w = Path(w)
            if (found := next(w.rglob("*.axm"), None)) is None:
                raise FileNotFoundError(f"在 {w} 中未找到 .axm 文件")

            ax_model = op.load(str(found))
            metadata = found.parent / "metadata.yaml"

        # ExecuTorch
        elif pte:
            LOGGER.info(f"为 ExecuTorch 推理加载 {w}...")
            # TorchAO 发布兼容性表错误 https://github.com/pytorch/ao/issues/2919
            check_requirements("setuptools<71.0.0")  # Setuptools 错误: https://github.com/pypa/setuptools/issues/4483
            check_requirements(("executorch==1.0.1", "flatbuffers"))
            from executorch.runtime import Runtime

            w = Path(w)
            if w.is_dir():
                model_file = next(w.rglob("*.pte"))
                metadata = w / "metadata.yaml"
            else:
                model_file = w
                metadata = w.parent / "metadata.yaml"

            program = Runtime.get().load_program(str(model_file))
            model = program.load_method("forward")

        # 任何其他格式 (不支持)
        else:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{w}' 不是支持的模型格式。Ultralytics 支持: {export_formats()['Format']}\n"
                f"请参见 https://docs.ultralytics.com/modes/predict 获取帮助。"
            )

        # 加载外部元数据 YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = YAML.load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch", "channels"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args"} and isinstance(v, str):
                    metadata[k] = ast.literal_eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
            kpt_names = metadata.get("kpt_names")
            end2end = metadata.get("args", {}).get("nms", False)
            dynamic = metadata.get("args", {}).get("dynamic", dynamic)
            ch = metadata.get("channels", 3)
        elif not (pt or triton or nn_module):
            LOGGER.warning(f"未找到 'model={w}' 的元数据")

        # 检查类别名称
        if "names" not in locals():  # 缺少类别名称
            names = default_class_names(data)
        names = check_class_names(names)

        self.__dict__.update(locals())  # 将所有变量分配给 self

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """在 AutoBackend 模型上运行推理

        参数:
            im (torch.Tensor): 要执行推理的图像张量
            augment (bool): 是否在推理期间执行数据增强
            visualize (bool): 是否可视化输出预测
            embed (list, optional): 要返回的特征向量/嵌入列表
            **kwargs (Any): 模型配置的额外关键字参数

        返回:
            (torch.Tensor | list[torch.Tensor]): 模型的原始输出张量
        """
        _b, _ch, h, w = im.shape  # batch、channel、height、width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # 转换为 FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW 转换为 numpy BHWC shape(1,320,192,3)

        # PyTorch
        if self.pt or self.nn_module:
            y = self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)

        # TorchScript
        elif self.jit:
            y = self.model(im)

        # ONNX OpenCV DNN
        elif self.dnn:
            im = im.cpu().numpy()  # torch 转换为 numpy
            self.net.setInput(im)
            y = self.net.forward()

        # ONNX Runtime
        elif self.onnx or self.imx:
            if self.use_io_binding:
                if not self.cuda:
                    im = im.cpu()
                self.io.bind_input(
                    name="images",
                    device_type=im.device.type,
                    device_id=im.device.index if im.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(im.shape),
                    buffer_ptr=im.data_ptr(),
                )
                self.session.run_with_iobinding(self.io)
                y = self.bindings
            else:
                im = im.cpu().numpy()  # torch 转换为 numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
            if self.imx:
                if self.task == "detect":
                    # boxes, conf, cls
                    y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
                elif self.task == "pose":
                    # boxes, conf, kpts
                    y = np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
                elif self.task == "segment":
                    y = (
                        np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                        y[4],
                    )

        # OpenVINO
        elif self.xml:
            im = im.cpu().numpy()  # FP32

            if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # 针对较大批次大小进行优化
                n = im.shape[0]  # 批次中的图像数量
                results = [None] * n  # 预分配与图像数量匹配的 None 列表

                def callback(request, userdata):
                    """使用 userdata 索引将结果放入预分配的列表中"""
                    results[userdata] = request.results

                # 创建 AsyncInferQueue,设置回调并为每个输入图像启动异步推理
                async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
                async_queue.set_callback(callback)
                for i in range(n):
                    # 使用 userdata=i 启动异步推理以指定结果列表中的位置
                    async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)  # 保持图像为 BCHW
                async_queue.wait_all()  # 等待所有推理请求完成
                y = [list(r.values()) for r in results]
                y = [np.concatenate(x) for x in zip(*y)]
            else:  # inference_mode = "LATENCY",针对 batch-size 1 的最快首次结果进行优化
                y = list(self.ov_compiled_model(im).values())

        # TensorRT
        elif self.engine:
            if self.dynamic and im.shape != self.bindings["images"].shape:
                if self.is_trt10:
                    self.context.set_input_shape("images", im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    i = self.model.get_binding_index("images")
                    self.context.set_binding_shape(i, im.shape)
                    self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        # CoreML
        elif self.coreml:
            im = im.cpu().numpy()
            if self.dynamic:
                im = im.transpose(0, 3, 1, 2)
            else:
                im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # 坐标是 xywh 归一化的
            if "confidence" in y:  # 包含 NMS
                from ultralytics.utils.ops import xywh2xyxy

                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy 像素
                cls = y["confidence"].argmax(1, keepdims=True)
                y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
            else:
                y = list(y.values())
            if len(y) == 2 and len(y[1].shape) != 4:  # 分割模型
                y = list(reversed(y))  # 对于分割模型反转 (pred, proto)

        # PaddlePaddle
        elif self.paddle:
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

        # MNN
        elif self.mnn:
            input_var = self.torch_to_mnn(im)
            output_var = self.net.onForward([input_var])
            y = [x.read() for x in output_var]

        # NCNN
        elif self.ncnn:
            mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
            with self.net.create_extractor() as ex:
                ex.input(self.net.input_names()[0], mat_in)
                # 警告: 'output_names' 排序作为 https://github.com/pnnx/pnnx/issues/130 的临时修复
                y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

        # NVIDIA Triton Inference Server
        elif self.triton:
            im = im.cpu().numpy()  # torch 转换为 numpy
            y = self.model(im)

        # RKNN
        elif self.rknn:
            im = (im.cpu().numpy() * 255).astype("uint8")
            im = im if isinstance(im, (list, tuple)) else [im]
            y = self.rknn_model.inference(inputs=im)

        # Axelera
        elif self.axelera:
            y = self.ax_model(im.cpu())

        # ExecuTorch
        elif self.pte:
            y = self.model.execute([im])

        # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
        else:
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model.serving_default(im)
                if not isinstance(y, list):
                    y = [y]
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite 或 Edge TPU
                details = self.input_details[0]
                is_int = details["dtype"] in {np.int8, np.int16}  # 是否为 TFLite 量化的 int8 或 int16 模型
                if is_int:
                    scale, zero_point = details["quantization"]
                    im = (im / scale + zero_point).astype(details["dtype"])  # 去量化
                self.interpreter.set_tensor(details["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if is_int:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # 重新量化
                    if x.ndim == 3:  # 如果任务不是分类,也排除掩码 (ndim=4)
                        # 根据图像大小反归一化 xywh。参见 https://github.com/ultralytics/ultralytics/pull/1695
                        # xywh 在 TFLite/EdgeTPU 中归一化以减轻整数模型的量化误差
                        if x.shape[-1] == 6 or self.end2end:  # 端到端模型
                            x[:, :, [0, 2]] *= w
                            x[:, :, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, :, 6::3] *= w
                                x[:, :, 7::3] *= h
                        else:
                            x[:, [0, 2]] *= w
                            x[:, [1, 3]] *= h
                            if self.task == "pose":
                                x[:, 5::3] *= w
                                x[:, 6::3] *= h
                    y.append(x)
            # TF 分割修复: 导出相对于 ONNX 导出是反转的,protos 被转置
            if len(y) == 2:  # 分割的 (det, proto) 输出顺序反转
                if len(y[1].shape) != 4:
                    y = list(reversed(y))  # 应该是 y = (1, 116, 8400), (1, 160, 160, 32)
                if y[1].shape[-1] == 6:  # 端到端模型
                    y = [y[1]]
                else:
                    y[1] = np.transpose(y[1], (0, 3, 1, 2))  # 应该是 y = (1, 116, 8400), (1, 32, 160, 160)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # 分割且未定义类别名称
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 32, 160, 160), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """将 NumPy 数组转换为模型设备上的 torch 张量

        参数:
            x (np.ndarray | torch.Tensor): 输入数组或张量

        返回:
            (torch.Tensor): 在 `self.device` 上的张量
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """通过使用虚拟输入运行一次前向传递来预热模型

        参数:
            imgsz (tuple[int, int, int, int]): (batch, channels, height, width) 格式的虚拟输入形状
        """
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # 预热模型
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 经验上 16 个框效果最好
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)  # 预热 NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt") -> list[bool]:
        """获取模型文件的路径并返回模型类型

        参数:
            p (str): 模型文件的路径

        返回:
            (list[bool]): 指示模型类型的布尔值列表

        示例:
            >>> types = AutoBackend._model_type("path/to/model.onnx")
            >>> assert types[2]  # onnx
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]  # 导出后缀
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)  # 检查
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")  # 保留对旧版 Apple CoreML *.mlmodel 格式的支持
        types[8] &= not types[9]  # tflite &= not edgetpu
        if any(types):
            triton = False
        else:
            from urllib.parse import urlsplit

            url = urlsplit(p)
            triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

        return [*types, triton]
