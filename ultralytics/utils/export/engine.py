"""
模型导出引擎模块 / Model Export Engine Module

本模块提供PyTorch模型到ONNX格式和TensorRT引擎的转换功能。
主要包含两个核心功能:
1. torch2onnx: 将PyTorch模型导出为ONNX格式
2. onnx2engine: 将ONNX模型转换为TensorRT引擎用于高性能推理

This module provides conversion functionality from PyTorch models to ONNX format and TensorRT engines.
Main features include:
1. torch2onnx: Export PyTorch models to ONNX format
2. onnx2engine: Convert ONNX models to TensorRT engines for high-performance inference
"""

from __future__ import annotations  # 启用延迟注解评估，支持类型提示中的前向引用 / Enable postponed annotations evaluation

import json  # 用于序列化和反序列化元数据 / For serializing and deserializing metadata
from pathlib import Path  # 用于跨平台路径操作 / For cross-platform path operations

import torch  # PyTorch深度学习框架 / PyTorch deep learning framework

from ultralytics.utils import IS_JETSON, LOGGER  # Jetson设备检测和日志工具 / Jetson device detection and logging utilities
from ultralytics.utils.torch_utils import TORCH_2_4  # PyTorch版本检测标志 / PyTorch version detection flag


def torch2onnx(
    torch_model: torch.nn.Module,
    im: torch.Tensor,
    onnx_file: str,
    opset: int = 14,
    input_names: list[str] = ["images"],
    output_names: list[str] = ["output0"],
    dynamic: bool | dict = False,
) -> None:
    """
    将PyTorch模型导出为ONNX格式。
    Export a PyTorch model to ONNX format.

    Args:
        torch_model (torch.nn.Module): 要导出的PyTorch模型 / The PyTorch model to export.
        im (torch.Tensor): 模型的示例输入张量 / Example input tensor for the model.
        onnx_file (str): 保存导出ONNX文件的路径 / Path to save the exported ONNX file.
        opset (int): 用于导出的ONNX操作集版本 / ONNX opset version to use for export.
        input_names (list[str]): 输入张量名称列表 / List of input tensor names.
        output_names (list[str]): 输出张量名称列表 / List of output tensor names.
        dynamic (bool | dict, optional): 是否启用动态轴 / Whether to enable dynamic axes.

    Notes:
        Setting `do_constant_folding=True` may cause issues with DNN inference for torch>=1.12.
        对于torch>=1.12，设置`do_constant_folding=True`可能会导致DNN推理问题。
    """
    # 如果是PyTorch 2.4+版本，需要禁用dynamo
    kwargs = {"dynamo": False} if TORCH_2_4 else {}
    torch.onnx.export(
        torch_model,
        im,
        onnx_file,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # 警告: torch>=1.12的DNN推理可能需要设置为False / WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None,
        **kwargs,
    )


def onnx2engine(
    onnx_file: str,
    engine_file: str | None = None,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> None:
    """
    将YOLO模型导出为TensorRT引擎格式。
    Export a YOLO model to TensorRT engine format.

    Args:
        onnx_file (str): 要转换的ONNX文件路径 / Path to the ONNX file to be converted.
        engine_file (str, optional): 保存生成的TensorRT引擎文件的路径 / Path to save the generated TensorRT engine file.
        workspace (int, optional): TensorRT的工作空间大小（单位：GB） / Workspace size in GB for TensorRT.
        half (bool, optional): 启用FP16精度 / Enable FP16 precision.
        int8 (bool, optional): 启用INT8精度 / Enable INT8 precision.
        dynamic (bool, optional): 启用动态输入形状 / Enable dynamic input shapes.
        shape (tuple[int, int, int, int], optional): 输入形状（批次，通道，高度，宽度） / Input shape (batch, channels, height, width).
        dla (int, optional): 要使用的DLA核心（仅限Jetson设备） / DLA core to use (Jetson devices only).
        dataset (ultralytics.data.build.InfiniteDataLoader, optional): 用于INT8校准的数据集 / Dataset for INT8 calibration.
        metadata (dict, optional): 要包含在引擎文件中的元数据 / Metadata to include in the engine file.
        verbose (bool, optional): 启用详细日志 / Enable verbose logging.
        prefix (str, optional): 日志消息的前缀 / Prefix for log messages.

    Raises:
        ValueError: 如果在非Jetson设备上启用DLA或未设置所需精度 / If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: 如果无法解析ONNX文件 / If the ONNX file cannot be parsed.

    Notes:
        TensorRT version compatibility is handled for workspace size and engine building.
        INT8 calibration requires a dataset and generates a calibration cache.
        Metadata is serialized and written to the engine file if provided.
        处理TensorRT版本兼容性以支持工作空间大小和引擎构建。
        INT8校准需要数据集并生成校准缓存。
        如果提供元数据，将序列化并写入引擎文件。
    """
    import tensorrt as trt  # 导入TensorRT库 / Import TensorRT library

    # 如果未指定引擎文件路径，使用ONNX文件路径替换后缀
    engine_file = engine_file or Path(onnx_file).with_suffix(".engine")

    # 创建TensorRT日志记录器
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # 创建引擎构建器 / Engine builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # 将工作空间大小从GB转换为字节
    workspace_bytes = int((workspace or 0) * (1 << 30))
    # 检测是否为TensorRT 10或更高版本
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10  # is TensorRT >= 10
    if is_trt10 and workspace_bytes > 0:
        # TensorRT 10+使用新的内存池API
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:  # TensorRT 7、8版本
        config.max_workspace_size = workspace_bytes
    # 设置显式批次标志
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    # 检查平台是否支持快速FP16/INT8并根据用户设置启用
    half = builder.platform_has_fast_fp16 and half
    int8 = builder.platform_has_fast_int8 and int8

    # 如果启用，可选地切换到DLA / Optionally switch to DLA if enabled
    if dla is not None:
        if not IS_JETSON:
            raise ValueError("DLA is only available on NVIDIA Jetson devices")
        LOGGER.info(f"{prefix} enabling DLA on core {dla}...")
        if not half and not int8:
            raise ValueError(
                "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
            )
        # 配置DLA设备类型和核心
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # 允许回退到GPU

    # 读取并解析ONNX文件 / Read ONNX file
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    # 获取并记录网络输入输出信息 / Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        profile = builder.create_optimization_profile()
        min_shape = (1, shape[1], 32, 32)  # minimum input shape
        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * d) for d in shape[2:]))  # max input shape
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8:
            config.set_calibration_profile(profile)

    LOGGER.info(f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {engine_file}")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        class EngineCalibrator(trt.IInt8Calibrator):
            """Custom INT8 calibrator for TensorRT engine optimization.

            This calibrator provides the necessary interface for TensorRT to perform INT8 quantization calibration using
            a dataset. It handles batch generation, caching, and calibration algorithm selection.

            Attributes:
                dataset: Dataset for calibration.
                data_iter: Iterator over the calibration dataset.
                algo (trt.CalibrationAlgoType): Calibration algorithm type.
                batch (int): Batch size for calibration.
                cache (Path): Path to save the calibration cache.

            Methods:
                get_algorithm: Get the calibration algorithm to use.
                get_batch_size: Get the batch size to use for calibration.
                get_batch: Get the next batch to use for calibration.
                read_calibration_cache: Use existing cache instead of calibrating again.
                write_calibration_cache: Write calibration cache to disk.
            """

            def __init__(
                self,
                dataset,  # ultralytics.data.build.InfiniteDataLoader
                cache: str = "",
            ) -> None:
                """Initialize the INT8 calibrator with dataset and cache path."""
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # DLA quantization needs ENTROPY_CALIBRATION_2
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION
                )
                self.batch = dataset.batch_size
                self.cache = Path(cache)

            def get_algorithm(self) -> trt.CalibrationAlgoType:
                """Get the calibration algorithm to use."""
                return self.algo

            def get_batch_size(self) -> int:
                """Get the batch size to use for calibration."""
                return self.batch or 1

            def get_batch(self, names) -> list[int] | None:
                """Get the next batch to use for calibration, as a list of device memory pointers."""
                try:
                    im0s = next(self.data_iter)["img"] / 255.0
                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    # Return None to signal to TensorRT there is no calibration data remaining
                    return None

            def read_calibration_cache(self) -> bytes | None:
                """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache: bytes) -> None:
                """Write calibration cache to disk."""
                _ = self.cache.write_bytes(cache)

        # Load dataset w/ builder (for batching) and calibrate
        config.int8_calibrator = EngineCalibrator(
            dataset=dataset,
            cache=str(Path(onnx_file).with_suffix(".cache")),
        )

    elif half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine, open(engine_file, "wb") as t:
        # Metadata
        if metadata is not None:
            meta = json.dumps(metadata)
            t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
            t.write(meta.encode())
        # Model
        t.write(engine if is_trt10 else engine.serialize())
