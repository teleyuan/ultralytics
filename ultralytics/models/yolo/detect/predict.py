"""
YOLO 目标检测预测模块

该模块实现了 YOLO 目标检测模型的预测功能,包括:
    - 后处理: NMS (非极大值抑制) 过滤重叠检测框
    - 结果构建: 将模型输出转换为 Results 对象
    - 目标特征提取: 从特征图中提取目标的特征向量

主要类:
    - DetectionPredictor: 目标检测预测器,继承自 BasePredictor

预测流程:
    1. 图像预处理 (调整大小、归一化)
    2. 模型前向传播
    3. 后处理 (NMS、坐标还原)
    4. 构建结果对象

典型应用:
    - 单张图像检测
    - 批量图像检测
    - 视频流检测
    - 实时检测
"""

from ultralytics.engine.predictor import BasePredictor  # 预测器基类
from ultralytics.engine.results import Results  # 结果对象
from ultralytics.utils import nms, ops  # NMS 和操作工具


class DetectionPredictor(BasePredictor):
    """扩展 BasePredictor 类的目标检测预测器类

    该预测器专门处理目标检测任务,将模型输出转换为包含边界框和类别预测的有意义的检测结果。

    属性:
        args (namespace): 预测器的配置参数
        model (nn.Module): 用于推理的检测模型
        batch (list): 用于处理的图像批次和元数据

    方法:
        postprocess: 将原始模型预测结果处理为检测结果
        construct_results: 从处理后的预测构建 Results 对象
        construct_result: 从单个预测创建一个 Result 对象
        get_obj_feats: 从特征图中提取目标特征

    示例:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """后处理预测结果并返回 Results 对象列表

        该方法对原始模型预测应用非极大值抑制,并为可视化和进一步分析做准备。

        对模型的原始预测进行后处理,包括:
            1. NMS (非极大值抑制) 过滤重叠检测框
            2. 坐标还原到原始图像尺寸
            3. 构建 Results 对象
            4. 提取目标特征 (如果需要)

        参数:
            preds (torch.Tensor): 模型的原始预测结果
                形状为 (batch, num_boxes, 85) 其中 85 = 4(bbox) + 1(conf) + 80(classes)
            img (torch.Tensor): 预处理后的输入图像张量 (模型输入格式)
            orig_imgs (torch.Tensor | list): 预处理前的原始输入图像
            **kwargs (Any): 额外的关键字参数

        返回:
            (list): 包含后处理预测结果的 Results 对象列表

        示例:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        # 检查是否需要保存目标特征
        save_feats = getattr(self, "_feats", None) is not None
        # 应用 NMS (非极大值抑制) 过滤重叠的检测框
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,  # 置信度阈值
            self.args.iou,  # IoU 阈值
            self.args.classes,  # 过滤的类别
            self.args.agnostic_nms,  # 类别无关的 NMS
            max_det=self.args.max_det,  # 最大检测数量
            nc=0 if self.args.task == "detect" else len(self.model.names),  # 类别数量
            end2end=getattr(self.model, "end2end", False),  # 端到端模型
            rotated=self.args.task == "obb",  # 旋转框检测
            return_idxs=save_feats,  # 返回索引 (用于特征提取)
        )

        # 如果输入图像是张量,则转换为 numpy 数组并调整通道顺序 (RGB -> BGR)
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor 而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        # 如果需要保存特征,则提取目标特征
        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        # 构建结果对象
        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        # 将目标特征添加到结果对象
        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # 添加目标特征到结果

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """从特征图中提取目标特征"""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # 找到最短的向量长度
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # 均值降维所有向量到相同长度
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # 对批次中的每张图像

    def construct_results(self, preds, img, orig_imgs):
        """从模型预测构建 Results 对象列表

        参数:
            preds (list[torch.Tensor]): 每张图像的预测边界框和分数列表
            img (torch.Tensor): 用于推理的预处理图像批次
            orig_imgs (list[np.ndarray]): 预处理前的原始图像列表

        返回:
            (list[Results]): 包含每张图像检测信息的 Results 对象列表
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """从单张图像预测构建一个 Results 对象

        参数:
            pred (torch.Tensor): 预测的边界框和分数,形状为 (N, 6),其中 N 是检测数量
            img (torch.Tensor): 用于推理的预处理图像张量
            orig_img (np.ndarray): 预处理前的原始图像
            img_path (str): 原始图像文件的路径

        返回:
            (Results): 包含原始图像、图像路径、类别名称和缩放后边界框的 Results 对象
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
