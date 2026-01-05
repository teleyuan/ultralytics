from ultralytics.models.yolo.segment import SegmentationValidator  # 导入分割验证器基类

class FastSAMValidator(SegmentationValidator):
    """
    FastSAM（Segment Anything Model，分割任意物体模型）自定义验证类，用于Ultralytics YOLO框架中的分割任务。

    该类继承自SegmentationValidator，专门为FastSAM定制了验证流程。此类将任务设置为'segment'，
    并使用SegmentMetrics进行评估。此外，为避免验证期间出现错误，绘图功能已被禁用。

    属性 (Attributes):
        dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器对象
        save_dir (Path): 验证结果保存的目录
        args (SimpleNamespace): 用于自定义验证过程的附加参数
        _callbacks (list): 验证期间要调用的回调函数列表
        metrics (SegmentMetrics): 用于评估的分割指标计算器

    方法 (Methods):
        __init__: 使用FastSAM的自定义设置初始化FastSAMValidator
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        初始化FastSAMValidator类，将任务设置为'segment'，指标设置为SegmentMetrics。

        参数 (Args):
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的DataLoader                                         
            save_dir (Path, optional): 保存结果的目录
            args (SimpleNamespace, optional): 验证器的配置
            _callbacks (list, optional): 验证期间要调用的回调函数列表

        注意 (Notes):
            此类中禁用了混淆矩阵和其他相关指标的绘图功能，以避免错误。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"  # 设置任务类型为分割
        self.args.plots = False  # 禁用混淆矩阵和其他绘图功能以避免错误
