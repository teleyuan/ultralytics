from copy import copy, deepcopy  # 导入copy模块，用于对象的浅拷贝和深拷贝

from ultralytics.models.yolo.segment import SegmentationTrainer  # 导入分割训练器基类
from ultralytics.nn.tasks import YOLOESegModel  # 导入YOLOE分割模型
from ultralytics.utils import RANK  # 导入RANK变量，用于分布式训练中标识进程

from .train import YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer  # 导入YOLOE检测训练器类
from .val import YOLOESegValidator  # 导入YOLOE分割验证器


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """YOLOE 分割模型的训练器类

    该类结合了 YOLOETrainer 和 SegmentationTrainer,专门为 YOLOE 分割模型提供训练功能,
    支持目标检测和实例分割能力。

    属性:
        cfg (dict): 包含训练参数的配置字典
        overrides (dict): 参数覆盖字典
        _callbacks (list): 训练事件的回调函数列表
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回使用指定配置和权重初始化的 YOLOESegModel

        参数:
            cfg (dict | str, optional): 模型配置字典或 YAML 文件路径
            weights (str, optional): 预训练权重文件的路径
            verbose (bool): 是否显示模型信息

        返回:
            (YOLOESegModel): 初始化的 YOLOE 分割模型
        """
        # 注意: 这里的 `nc` 是一张图像中不同文本样本的最大数量,而不是实际的 `nc`
        # 注意: 按照官方配置,nc 目前硬编码为 80
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """创建并返回用于 YOLOE 分割模型评估的验证器

        返回:
            (YOLOESegValidator): YOLOE 分割模型的验证器
        """
        self.loss_names = "box", "seg", "cls", "dfl"
        return YOLOESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class YOLOEPESegTrainer(SegmentationTrainer):
    """以线性探测方式微调 YOLOESeg 模型

    该训练器专门使用线性探测方法微调 YOLOESeg 模型,该方法涉及冻结模型的大部分层,
    仅训练特定层以有效适应新任务。

    属性:
        data (dict): 包含通道、类别名称和类别数量的数据集配置
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回为线性探测配置的使用指定配置和权重初始化的 YOLOESegModel

        参数:
            cfg (dict | str, optional): 模型配置字典或 YAML 文件路径
            weights (str, optional): 预训练权重文件的路径
            verbose (bool): 是否显示模型信息

        返回:
            (YOLOESegModel): 为线性探测配置的初始化 YOLOE 分割模型
        """
        # 注意: 这里的 `nc` 是一张图像中不同文本样本的最大数量,而不是实际的 `nc`
        # 注意: 按照官方配置,nc 目前硬编码为 80
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "线性探测必须提供预训练权重。"
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # 注意: `get_text_pe` 与文本模型和 YOLOEDetect.reprta 相关,
        # 只要加载正确的预训练权重就能得到正确结果
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.pe
        model.train()

        return model


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """从零开始训练 YOLOE 分割模型的训练器,无需预训练权重"""

    pass


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """具有视觉提示 (VP) 能力的 YOLOE 分割模型训练器"""

    pass
