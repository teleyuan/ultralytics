"""
模型验证模块

该模块提供在数据集的测试集或验证集上检查模型精度的功能。
支持多种模型格式的验证，并计算各种评估指标。

使用示例:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

使用示例 - 模型格式:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

# 标准库导入
import json  # JSON 数据处理
import time  # 时间相关函数
from pathlib import Path  # 跨平台路径操作

# 第三方库导入
import numpy as np  # 数组和数值计算
import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # 分布式训练

# Ultralytics 模块导入
from ultralytics.cfg import get_cfg, get_save_dir  # 配置管理
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 数据集检查
from ultralytics.nn.autobackend import AutoBackend  # 自动后端加载器
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis  # 工具函数
from ultralytics.utils.checks import check_imgsz  # 检查函数
from ultralytics.utils.ops import Profile  # 性能分析工具
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model  # PyTorch 工具


class BaseValidator:
    """用于创建验证器的基类。

    该类为验证过程提供基础功能，包括模型评估、指标计算和结果可视化。

    属性:
        args (SimpleNamespace): 验证器的配置。
        dataloader (DataLoader): 用于验证的数据加载器。
        model (nn.Module): 要验证的模型。
        data (dict): 包含数据集信息的数据字典。
        device (torch.device): 用于验证的设备。
        batch_i (int): 当前批次索引。
        training (bool): 模型是否处于训练模式。
        names (dict): 类别名称映射。
        seen (int): 验证期间已处理的图像数量。
        stats (dict): 验证期间收集的统计信息。
        confusion_matrix: 用于分类评估的混淆矩阵。
        nc (int): 类别数量。
        iouv (torch.Tensor): IoU 阈值，从 0.50 到 0.95，间隔为 0.05。
        jdict (list): 用于存储 JSON 验证结果的列表。
        speed (dict): 包含 'preprocess'、'inference'、'loss'、'postprocess' 键及其各自批次
            处理时间（毫秒）的字典。
        save_dir (Path): 保存结果的目录。
        plots (dict): 用于存储可视化图表的字典。
        callbacks (dict): 用于存储各种回调函数的字典。
        stride (int): 用于填充计算的模型步长。
        loss (torch.Tensor): 训练验证期间累积的损失。

    方法:
        __call__: 执行验证过程，在数据加载器上运行推理并计算性能指标。
        match_predictions: 使用 IoU 将预测结果与真实对象匹配。
        add_callback: 将给定的回调函数附加到指定事件。
        run_callbacks: 运行与指定事件关联的所有回调函数。
        get_dataloader: 从数据集路径和批次大小获取数据加载器。
        build_dataset: 从图像路径构建数据集。
        preprocess: 预处理输入批次。
        postprocess: 后处理预测结果。
        init_metrics: 初始化 YOLO 模型的性能指标。
        update_metrics: 基于预测结果和批次更新指标。
        finalize_metrics: 完成并返回所有指标。
        get_stats: 返回模型性能的统计信息。
        print_results: 打印模型预测的结果。
        get_desc: 获取 YOLO 模型的描述。
        on_plot: 注册用于可视化的图表。
        plot_val_samples: 在训练期间绘制验证样本。
        plot_predictions: 在批次图像上绘制 YOLO 模型预测。
        pred_to_json: 将预测结果转换为 JSON 格式。
        eval_json: 评估并返回预测统计信息的 JSON 格式。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """初始化 BaseValidator 实例。

        参数:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (Path, optional): 保存结果的目录。
            args (SimpleNamespace, optional): 验证器的配置。
            _callbacks (dict, optional): 用于存储各种回调函数的字典。
        """
        import torchvision  # noqa (在这里导入，以便 torchvision 导入时间不记录在后处理时间中)

        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.01 if self.args.task == "obb" else 0.001  # 降低 OBB 验证的内存使用
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证过程，在数据加载器上运行推理并计算性能指标。

        参数:
            trainer (object, optional): 包含要验证模型的训练器对象。
            model (nn.Module, optional): 如果不使用训练器，要验证的模型。

        返回:
            (dict): 包含验证统计信息的字典。
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # 训练期间强制使用 FP16 验证
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # 验证未编译的原始模型以避免问题
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py 导出的模型默认批次大小为 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # CPU 验证更快，因为时间主要用于推理而非数据加载
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # 在 get_dataloader() 中用于填充
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # 预热

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # 每次验证前清空
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # 预处理
            with dt[0]:
                batch = self.preprocess(batch)

            # 推理
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # 损失计算
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # 后处理
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            # 在所有 GPU 上聚合损失
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # 返回结果保留 5 位小数
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # 展平并保存
                stats = self.eval_json(stats)  # 更新统计信息
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """使用 IoU 将预测结果与真实对象匹配。

        参数:
            pred_classes (torch.Tensor): 形状为 (N,) 的预测类别索引。
            true_classes (torch.Tensor): 形状为 (M,) 的真实类别索引。
            iou (torch.Tensor): 包含预测和真实对象之间成对 IoU 值的 NxM 张量。
            use_scipy (bool, optional): 是否使用 scipy 进行匹配（更精确）。

        返回:
            (torch.Tensor): 形状为 (N, 10) 的正确张量，对应 10 个 IoU 阈值。
        """
        # Dx10 矩阵，其中 D - 检测数量，10 - IoU 阈值
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD 矩阵，其中 L - 标签（行），D - 检测（列）
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # 将错误类别的 IoU 置零
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # 警告: 已知问题会降低 mAP，参见 https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # 限定导入范围以避免为所有命令导入

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > 阈值且类别匹配
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """将给定的回调函数附加到指定事件。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """运行与指定事件关联的所有回调函数。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """从数据集路径和批次大小获取数据加载器。"""
        raise NotImplementedError("此验证器未实现 get_dataloader 函数")

    def build_dataset(self, img_path):
        """从图像路径构建数据集。"""
        raise NotImplementedError("验证器中未实现 build_dataset 函数")

    def preprocess(self, batch):
        """预处理输入批次。"""
        return batch

    def postprocess(self, preds):
        """后处理预测结果。"""
        return preds

    def init_metrics(self, model):
        """初始化 YOLO 模型的性能指标。"""
        pass

    def update_metrics(self, preds, batch):
        """基于预测结果和批次更新指标。"""
        pass

    def finalize_metrics(self):
        """完成并返回所有指标。"""
        pass

    def get_stats(self):
        """返回模型性能的统计信息。"""
        return {}

    def gather_stats(self):
        """在 DDP 训练期间从所有 GPU 收集统计信息到 GPU 0。"""
        pass

    def print_results(self):
        """打印模型预测的结果。"""
        pass

    def get_desc(self):
        """获取 YOLO 模型的描述。"""
        pass

    @property
    def metric_keys(self):
        """返回 YOLO 训练/验证中使用的指标键。"""
        return []

    def on_plot(self, name, data=None):
        """注册用于可视化的图表，按类型去重。"""
        plot_type = data.get("type") if data else None
        if plot_type and any((v.get("data") or {}).get("type") == plot_type for v in self.plots.values()):
            return  # 跳过重复的图表类型
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        """在训练期间绘制验证样本。"""
        pass

    def plot_predictions(self, batch, preds, ni):
        """在批次图像上绘制 YOLO 模型预测。"""
        pass

    def pred_to_json(self, preds, batch):
        """将预测结果转换为 JSON 格式。"""
        pass

    def eval_json(self, stats):
        """评估并返回预测统计信息的 JSON 格式。"""
        pass
