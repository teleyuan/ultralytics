from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """实现 HungarianMatcher 的模块，用于预测和真实值之间的最优分配。

    HungarianMatcher 使用成本函数对预测边界框和真实边界框执行最优二分匹配，
    该成本函数考虑分类分数、边界框坐标以及可选的掩码预测。
    这用于像 DETR 这样的端到端目标检测模型。

    属性:
        cost_gain (dict[str, float]): 'class'、'bbox'、'giou'、'mask' 和 'dice' 组件的成本系数字典。
        use_fl (bool): 是否使用 Focal Loss 计算分类成本。
        with_mask (bool): 模型是否进行掩码预测。
        num_sample_points (int): 掩码成本计算中使用的采样点数量。
        alpha (float): Focal Loss 计算中的 alpha 因子。
        gamma (float): Focal Loss 计算中的 gamma 因子。

    方法:
        forward: 计算批次中预测和真实值之间的最优分配。
        _cost_mask: 如果预测掩码，则计算掩码成本和 dice 成本。

    示例:
        使用自定义成本增益初始化 HungarianMatcher
        >>> matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})

        执行预测和真实值之间的匹配
        >>> pred_boxes = torch.rand(2, 100, 4)  # batch_size=2, num_queries=100
        >>> pred_scores = torch.rand(2, 100, 80)  # 80 个类别
        >>> gt_boxes = torch.rand(10, 4)  # 10 个真实边界框
        >>> gt_classes = torch.randint(0, 80, (10,))
        >>> gt_groups = [5, 5]  # 每张图像 5 个真实框
        >>> indices = matcher(pred_boxes, pred_scores, gt_boxes, gt_classes, gt_groups)
    """

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """初始化 HungarianMatcher 以实现预测边界框和真实边界框的最优分配。

        参数:
            cost_gain (dict[str, float], optional): 不同匹配成本组件的成本系数字典。
                应包含 'class'、'bbox'、'giou'、'mask' 和 'dice' 键。
            use_fl (bool): 是否使用 Focal Loss 计算分类成本。
            with_mask (bool): 模型是否进行掩码预测。
            num_sample_points (int): 掩码成本计算中使用的采样点数量。
            alpha (float): Focal Loss 计算中的 alpha 因子。
            gamma (float): Focal Loss 计算中的 gamma 因子。
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """使用匈牙利算法计算预测和真实值之间的最优分配。

        该方法基于分类分数、边界框坐标以及可选的掩码预测计算匹配成本，
        然后找到预测和真实值之间的最优二分分配。

        参数:
            pred_bboxes (torch.Tensor): 预测的边界框，形状为 (batch_size, num_queries, 4)。
            pred_scores (torch.Tensor): 预测的分类分数，形状为 (batch_size, num_queries, num_classes)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (num_gts, 4)。
            gt_cls (torch.Tensor): 真实类别标签，形状为 (num_gts,)。
            gt_groups (list[int]): 批次中每张图像的真实框数量。
            masks (torch.Tensor, optional): 预测的掩码，形状为 (batch_size, num_queries, height, width)。
            gt_mask (list[torch.Tensor], optional): 真实掩码，每个形状为 (num_masks, Height, Width)。

        返回:
            (list[tuple[torch.Tensor, torch.Tensor]]): 大小为 batch_size 的列表，每个元素是一个元组 (index_i, index_j)，
                其中 index_i 是所选预测的索引张量（按顺序），index_j 是相应所选真实目标的索引张量（按顺序）。
            对于每个批次元素，满足: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)。
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # 展平以批量格式计算成本矩阵
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # 计算分类成本
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # 计算边界框之间的 L1 成本
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # 计算边界框之间的 GIoU 成本，(bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # 将成本组合成最终成本矩阵
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )

        # 如果可用，添加掩码成本
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # 将无效值（NaN 和无穷大）设置为 0
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (查询的索引, 真实值的索引)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # 此函数用于未来的 RT-DETR 分割模型
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.amp.autocast("cuda", enabled=False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C


def get_cdn_group(
    batch: dict[str, Any],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
    """从真实值生成包含正样本和负样本的对比去噪训练组。

    该函数通过向真实边界框和类别标签添加噪声来创建对比去噪训练的去噪查询。
    它生成正样本和负样本以提高模型的鲁棒性。

    参数:
        batch (dict[str, Any]): 批次字典，包含 'gt_cls'（形状为 (num_gts,) 的 torch.Tensor）、
            'gt_bboxes'（形状为 (num_gts, 4) 的 torch.Tensor）以及 'gt_groups'（list[int]），
            表示每张图像的真实对象数量。
        num_classes (int): 对象类别总数。
        num_queries (int): 对象查询数量。
        class_embed (torch.Tensor): 类别嵌入权重，用于将标签映射到嵌入空间。
        num_dn (int): 要生成的去噪查询数量。
        cls_noise_ratio (float): 类别标签的噪声比率。
        box_noise_scale (float): 边界框坐标的噪声比例。
        training (bool): 模型是否处于训练模式。

    返回:
        padding_cls (torch.Tensor | None): 用于去噪的修改后类别嵌入，形状为 (bs, num_dn, embed_dim)。
        padding_bbox (torch.Tensor | None): 用于去噪的修改后边界框，形状为 (bs, num_dn, 4)。
        attn_mask (torch.Tensor | None): 用于去噪的注意力掩码，形状为 (tgt_size, tgt_size)。
        dn_meta (dict[str, Any] | None): 包含去噪参数的元信息字典。

    示例:
        为训练生成去噪组
        >>> batch = {
        ...     "cls": torch.tensor([0, 1, 2]),
        ...     "bboxes": torch.rand(3, 4),
        ...     "batch_idx": torch.tensor([0, 0, 1]),
        ...     "gt_groups": [2, 1],
        ... }
        >>> class_embed = torch.rand(80, 256)  # 80 个类别，256 维嵌入
        >>> cdn_outputs = get_cdn_group(batch, 80, 100, class_embed, training=True)
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # 将真实值填充到批次的最大数量
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]

    # 每个组有正查询和负查询
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # 正负掩码
    # (bs*num*num_group, ), 第二个 total_num*num_group 部分作为负样本
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # 对一半样本应用类别标签噪声
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # 随机分配新的类别标签
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # 反 sigmoid

    num_dn = int(max_nums * 2 * num_group)  # 总去噪查询数
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # 匹配查询不能看到重建查询
    attn_mask[num_dn:, :num_dn] = True
    # 重建查询之间不能互相看到
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
