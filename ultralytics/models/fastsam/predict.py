import torch  # PyTorch深度学习框架
from PIL import Image  # Python图像处理库，用于图像操作

from ultralytics.models.yolo.segment import SegmentationPredictor  # 导入分割预测器基类
from ultralytics.utils import DEFAULT_CFG  # 导入默认配置
from ultralytics.utils.metrics import box_iou  # 导入边界框IoU计算函数
from ultralytics.utils.ops import scale_masks  # 导入掩码缩放函数
from ultralytics.utils.torch_utils import TORCH_1_10  # 导入PyTorch版本标志

from .utils import adjust_bboxes_to_image_border  # 导入边界框调整工具函数


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAM预测器类，专门用于快速SAM（Segment Anything Model，分割任意物体模型）分割预测任务。

    该类继承自SegmentationPredictor，针对FastSAM定制了预测流程。它调整了后处理步骤，
    整合了掩码预测和非极大值抑制，同时针对单类分割进行了优化。

    属性:
        prompts (dict): 包含分割提示信息的字典（边界框、点、标签、文本）  
        device (torch.device): 模型和张量处理所使用的设备              
        clip (Any, optional): 用于基于文本提示的CLIP模型，按需加载                

    方法 (Methods):
        postprocess: 对FastSAM预测结果应用后处理并处理提示
        prompt: 基于各种提示类型执行图像分割推理
        set_prompts: 设置推理期间使用的提示
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        使用配置和回调函数初始化FastSAM预测器。

        初始化一个专门用于Fast SAM（Segment Anything Model，分割任意物体模型）分割任务的预测器。
        该预测器继承自SegmentationPredictor，并具有针对单类分割优化的掩码预测和非极大值抑制的自定义后处理。

        参数 (Args):
            cfg (dict): 预测器的配置
            overrides (dict, optional): 配置覆盖项
            _callbacks (list, optional): 回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}  # 初始化提示字典

    def postprocess(self, preds, img, orig_imgs):
        """
        对FastSAM预测结果应用后处理并处理提示。

        参数 (Args):
            preds (list[torch.Tensor]): 模型的原始预测结果
            img (torch.Tensor): 输入到模型的图像张量
            orig_imgs (list[np.ndarray]): 预处理前的原始图像

        返回 (Returns):
            (list[Results]): 应用提示后的处理结果
        """
        # 从提示字典中提取各种提示信息
        bboxes = self.prompts.pop("bboxes", None)  # 边界框提示
        points = self.prompts.pop("points", None)  # 点提示
        labels = self.prompts.pop("labels", None)  # 标签提示
        texts = self.prompts.pop("texts", None)  # 文本提示

        # 调用父类的后处理方法
        results = super().postprocess(preds, img, orig_imgs)

        # 对每个结果进行处理
        for result in results:
            # 创建完整图像边界框 [左上角x, 左上角y, 右下角x, 右下角y]
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            # 调整边界框使其贴合图像边界
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            # 找出与完整边界框IoU大于0.9的边界框索引
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                # 将这些边界框替换为完整边界框
                result.boxes.xyxy[idx] = full_box

        # 应用提示并返回结果
        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        基于边界框、点和文本提示等线索执行图像分割推理。

        参数 (Args):
            results (Results | list[Results]): FastSAM模型未应用任何提示的原始推理结果    
            bboxes (np.ndarray | list, optional): 边界框，形状为(N, 4)，XYXY格式
            points (np.ndarray | list, optional): 表示物体位置的点，形状为(N, 2)，以像素为单位
            labels (np.ndarray | list, optional): 点提示的标签，形状为(N,)，1表示前景，0表示背景
            texts (str | list[str], optional): 文本提示，包含字符串对象的列表

        返回 (Returns):
            (list[Results]): 根据提供的提示过滤和确定的输出结果
        """
        # 如果没有提供任何提示，直接返回原始结果
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        # 确保results是列表格式
        if not isinstance(results, list):
            results = [results]
        for result in results:
            # 如果结果为空，直接添加并跳过
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            # 如果掩码尺寸与原始图像尺寸不匹配，进行缩放
            if masks.shape[1:] != result.orig_shape:
                masks = (scale_masks(masks[None].float(), result.orig_shape)[0] > 0.5).byte()

            # 处理边界框提示 (bboxes prompt)
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                # 将边界框转换为张量
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                # 确保边界框是二维的
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                # 计算边界框面积
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                # 计算每个边界框内的掩码面积
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                # 计算完整掩码面积
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                # 计算并集面积（边界框面积 + 完整掩码面积 - 交集面积）
                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                # 选择IoU最大的掩码
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            # 处理点提示 (points prompt)
            if points is not None:
                # 将点转换为张量
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                # 确保点是二维的
                points = points[None] if points.ndim == 1 else points
                # 如果没有提供标签，默认全部为前景点（标签为1）
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                # 验证点和标签的数量匹配
                assert len(labels) == len(points), (
                    f"Expected `labels` to have the same length as `points`, but got {len(labels)} and {len(points)}."
                )
                # 初始化点索引：如果全是负样本点，初始化为全True，否则为全False
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # 所有点都是负样本点
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                # 遍历每个点和对应的标签
                for point, label in zip(points, labels):
                    # 找出包含该点的掩码，并根据标签设置索引
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                # 将点索引与边界框索引合并（OR操作）
                idx |= point_idx
            # 处理文本提示 (text prompt)
            if texts is not None:
                # 确保文本是列表格式
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                # 遍历每个边界框，裁剪对应的图像区域
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    # 过滤掉掩码面积过小的区域（torch 1.9的bug修复）
                    if (masks[i].sum() if TORCH_1_10 else masks[i].sum(0).sum()) <= 100:  # torch 1.9 bug workaround
                        filter_idx.append(i)
                        continue
                    # 裁剪图像并转换为PIL格式（注意BGR到RGB的转换）
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                # 使用CLIP模型计算裁剪图像与文本之间的相似度
                similarity = self._clip_inference(crop_ims, texts)
                # 选择相似度最高的索引
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                # 调整索引以考虑被过滤的项
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            # 根据索引筛选结果
            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """
        执行CLIP推理以计算图像和文本提示之间的相似度。

        参数 (Args):
            images (list[PIL.Image]): 源图像列表，每个应为RGB通道顺序的PIL.Image
            texts (list[str]): 文本提示列表，每个应为字符串对象

        返回 (Returns):
            (torch.Tensor): 给定图像和文本之间的相似度矩阵，形状为(M, N)
        """
        from ultralytics.nn.text_model import CLIP  # 导入CLIP模型

        # 如果CLIP模型未初始化，则创建新实例
        if not hasattr(self, "clip"):
            self.clip = CLIP("ViT-B/32", device=self.device)
        # 预处理图像并堆叠成批次
        images = torch.stack([self.clip.image_preprocess(image).to(self.device) for image in images])
        # 提取图像特征
        image_features = self.clip.encode_image(images)
        # 提取文本特征
        text_features = self.clip.encode_text(self.clip.tokenize(texts))
        # 计算文本和图像特征的相似度矩阵（矩阵乘法）
        return text_features @ image_features.T  # (M, N)

    def set_prompts(self, prompts):
        """
        设置推理期间使用的提示。

        参数 (Args):
            prompts (dict): 包含各种提示信息的字典
        """
        self.prompts = prompts
