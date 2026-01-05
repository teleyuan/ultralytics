"""
Ultralytics 基础回调函数模块。

Base callbacks for Ultralytics training, validation, prediction, and export processes.

本模块定义了 Ultralytics 训练、验证、预测和导出过程中的基础回调函数。
这些回调函数在模型训练生命周期的不同阶段被调用，允许用户在特定时刻插入自定义逻辑，
例如记录指标、保存模型、可视化等。所有回调函数都是空实现，供各种集成（如 TensorBoard、MLflow 等）继承和扩展。
"""

from collections import defaultdict  # 默认字典，用于创建回调字典
from copy import deepcopy  # 深拷贝工具，用于复制回调字典

# Trainer callbacks ----------------------------------------------------------------------------------------------------
# 训练器回调函数 - 在训练过程的不同阶段被调用


def on_pretrain_routine_start(trainer):
    """
    预训练例程开始前调用。

    Called before the pretraining routine starts.

    该回调在预训练例程开始之前触发，可用于初始化日志记录器、设置环境等。

    Args:
        trainer: 训练器实例，包含训练配置和状态信息
    """
    pass


def on_pretrain_routine_end(trainer):
    """
    预训练例程结束后调用。

    Called after the pretraining routine ends.

    该回调在预训练例程完成后触发，可用于记录初始化信息、保存配置等。

    Args:
        trainer: 训练器实例，包含训练配置和状态信息
    """
    pass


def on_train_start(trainer):
    """
    训练开始时调用。

    Called when the training starts.

    该回调在训练循环开始时触发，可用于记录模型架构、初始参数等。

    Args:
        trainer: 训练器实例，包含训练配置和状态信息
    """
    pass


def on_train_epoch_start(trainer):
    """
    每个训练周期开始时调用。

    Called at the start of each training epoch.

    该回调在每个训练周期（epoch）开始时触发，可用于调整学习率、记录周期信息等。

    Args:
        trainer: 训练器实例，包含当前周期的训练状态
    """
    pass


def on_train_batch_start(trainer):
    """
    每个训练批次开始时调用。

    Called at the start of each training batch.

    该回调在每个训练批次（batch）开始时触发，可用于数据预处理、批次级别的日志记录等。

    Args:
        trainer: 训练器实例，包含当前批次的训练状态
    """
    pass


def optimizer_step(trainer):
    """
    优化器执行更新步骤时调用。

    Called when the optimizer takes a step.

    该回调在优化器更新模型参数时触发，可用于梯度裁剪、自定义优化逻辑等。

    Args:
        trainer: 训练器实例，包含优化器和模型状态
    """
    pass


def on_before_zero_grad(trainer):
    """
    梯度清零前调用。

    Called before the gradients are set to zero.

    该回调在梯度清零操作之前触发，可用于梯度分析、梯度累积等。

    Args:
        trainer: 训练器实例，包含当前梯度信息
    """
    pass


def on_train_batch_end(trainer):
    """
    每个训练批次结束时调用。

    Called at the end of each training batch.

    该回调在每个训练批次完成后触发，可用于批次级别的指标记录、中间结果保存等。

    Args:
        trainer: 训练器实例，包含批次训练结果
    """
    pass


def on_train_epoch_end(trainer):
    """
    每个训练周期结束时调用。

    Called at the end of each training epoch.

    该回调在每个训练周期完成后触发，可用于记录训练损失、学习率等周期级别的指标。

    Args:
        trainer: 训练器实例，包含周期训练统计信息
    """
    pass


def on_fit_epoch_end(trainer):
    """
    每个拟合周期结束时调用（训练 + 验证）。

    Called at the end of each fit epoch (train + val).

    该回调在每个完整周期（包括训练和验证）结束后触发，可用于记录综合指标、模型检查点等。

    Args:
        trainer: 训练器实例，包含训练和验证的综合结果
    """
    pass


def on_model_save(trainer):
    """
    模型保存时调用。

    Called when the model is saved.

    该回调在模型权重保存时触发，可用于记录检查点信息、上传模型等。

    Args:
        trainer: 训练器实例，包含要保存的模型信息
    """
    pass


def on_train_end(trainer):
    """
    训练结束时调用。

    Called when the training ends.

    该回调在整个训练过程完成后触发，可用于记录最终结果、清理资源、生成报告等。

    Args:
        trainer: 训练器实例，包含完整的训练历史和最终结果
    """
    pass


def on_params_update(trainer):
    """
    模型参数更新时调用。

    Called when the model parameters are updated.

    该回调在模型参数发生更新时触发，可用于参数监控、动态调整等。

    Args:
        trainer: 训练器实例，包含更新后的模型参数
    """
    pass


def teardown(trainer):
    """
    训练过程拆解清理时调用。

    Called during the teardown of the training process.

    该回调在训练过程清理阶段触发，可用于关闭日志记录器、释放资源、清理临时文件等。

    Args:
        trainer: 训练器实例，包含需要清理的资源信息
    """
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------
# 验证器回调函数 - 在验证过程的不同阶段被调用


def on_val_start(validator):
    """
    验证开始时调用。

    Called when the validation starts.

    该回调在验证过程开始时触发，可用于初始化验证指标、记录验证配置等。

    Args:
        validator: 验证器实例，包含验证配置和数据集信息
    """
    pass


def on_val_batch_start(validator):
    """
    每个验证批次开始时调用。

    Called at the start of each validation batch.

    该回调在每个验证批次开始时触发，可用于批次级别的预处理和日志记录。

    Args:
        validator: 验证器实例，包含当前批次的验证状态
    """
    pass


def on_val_batch_end(validator):
    """
    每个验证批次结束时调用。

    Called at the end of each validation batch.

    该回调在每个验证批次结束后触发，可用于批次级别的指标计算和中间结果保存。

    Args:
        validator: 验证器实例，包含批次验证结果
    """
    pass


def on_val_end(validator):
    """
    验证结束时调用。

    Called when the validation ends.

    该回调在验证过程完成后触发，可用于记录最终验证指标、生成验证报告等。

    Args:
        validator: 验证器实例，包含完整的验证结果和统计信息
    """
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------
# 预测器回调函数 - 在预测过程的不同阶段被调用


def on_predict_start(predictor):
    """
    预测开始时调用。

    Called when the prediction starts.

    该回调在预测过程开始时触发，可用于初始化预测环境、记录预测配置等。

    Args:
        predictor: 预测器实例，包含预测配置和模型信息
    """
    pass


def on_predict_batch_start(predictor):
    """
    每个预测批次开始时调用。

    Called at the start of each prediction batch.

    该回调在每个预测批次开始时触发，可用于批次级别的数据预处理。

    Args:
        predictor: 预测器实例，包含当前批次的预测输入
    """
    pass


def on_predict_batch_end(predictor):
    """
    每个预测批次结束时调用。

    Called at the end of each prediction batch.

    该回调在每个预测批次完成后触发，可用于批次级别的结果处理。

    Args:
        predictor: 预测器实例，包含批次预测的原始输出
    """
    pass


def on_predict_postprocess_end(predictor):
    """
    预测后处理结束后调用。

    Called after the post-processing of the prediction ends.

    该回调在预测结果后处理完成后触发，可用于记录处理后的结果、可视化等。

    Args:
        predictor: 预测器实例，包含后处理后的预测结果
    """
    pass


def on_predict_end(predictor):
    """
    预测结束时调用。

    Called when the prediction ends.

    该回调在整个预测过程完成后触发，可用于保存所有预测结果、生成报告、清理资源等。

    Args:
        predictor: 预测器实例，包含所有预测结果和统计信息
    """
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------
# 导出器回调函数 - 在模型导出过程的不同阶段被调用


def on_export_start(exporter):
    """
    模型导出开始时调用。

    Called when the model export starts.

    该回调在模型导出过程开始时触发，可用于记录导出配置、初始化导出环境等。

    Args:
        exporter: 导出器实例，包含导出配置和模型信息
    """
    pass


def on_export_end(exporter):
    """
    模型导出结束时调用。

    Called when the model export ends.

    该回调在模型导出完成后触发，可用于验证导出结果、记录导出信息、上传导出的模型等。

    Args:
        exporter: 导出器实例，包含导出结果和文件路径信息
    """
    pass


# 默认回调字典 - 包含所有生命周期阶段的回调函数映射
default_callbacks = {
    # Run in trainer
    # 在训练器中运行的回调函数
    "on_pretrain_routine_start": [on_pretrain_routine_start],  # 预训练例程开始
    "on_pretrain_routine_end": [on_pretrain_routine_end],  # 预训练例程结束
    "on_train_start": [on_train_start],  # 训练开始
    "on_train_epoch_start": [on_train_epoch_start],  # 训练周期开始
    "on_train_batch_start": [on_train_batch_start],  # 训练批次开始
    "optimizer_step": [optimizer_step],  # 优化器步骤
    "on_before_zero_grad": [on_before_zero_grad],  # 梯度清零前
    "on_train_batch_end": [on_train_batch_end],  # 训练批次结束
    "on_train_epoch_end": [on_train_epoch_end],  # 训练周期结束
    "on_fit_epoch_end": [on_fit_epoch_end],  # 拟合周期结束（训练 + 验证）
    "on_model_save": [on_model_save],  # 模型保存
    "on_train_end": [on_train_end],  # 训练结束
    "on_params_update": [on_params_update],  # 参数更新
    "teardown": [teardown],  # 清理拆解
    # Run in validator
    # 在验证器中运行的回调函数
    "on_val_start": [on_val_start],  # 验证开始
    "on_val_batch_start": [on_val_batch_start],  # 验证批次开始
    "on_val_batch_end": [on_val_batch_end],  # 验证批次结束
    "on_val_end": [on_val_end],  # 验证结束
    # Run in predictor
    # 在预测器中运行的回调函数
    "on_predict_start": [on_predict_start],  # 预测开始
    "on_predict_batch_start": [on_predict_batch_start],  # 预测批次开始
    "on_predict_postprocess_end": [on_predict_postprocess_end],  # 预测后处理结束
    "on_predict_batch_end": [on_predict_batch_end],  # 预测批次结束
    "on_predict_end": [on_predict_end],  # 预测结束
    # Run in exporter
    # 在导出器中运行的回调函数
    "on_export_start": [on_export_start],  # 导出开始
    "on_export_end": [on_export_end],  # 导出结束
}


def get_default_callbacks():
    """
    获取 Ultralytics 训练、验证、预测和导出过程的默认回调函数。

    Get the default callbacks for Ultralytics training, validation, prediction, and export processes.

    该函数返回一个包含所有默认回调函数的字典副本。字典使用 defaultdict 结构，
    这样即使访问不存在的键也会返回空列表而不是抛出异常。使用深拷贝确保返回的是
    独立的副本，避免多个实例之间共享回调列表。

    Returns:
        (dict): 包含各种训练事件默认回调函数的字典。每个键代表训练过程中的一个事件，
            对应的值是一个回调函数列表，这些函数会在该事件发生时按顺序执行。

    Examples:
        >>> callbacks = get_default_callbacks()
        >>> print(list(callbacks.keys()))  # 显示所有可用的回调事件
        ['on_pretrain_routine_start', 'on_pretrain_routine_end', ...]
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """
    向实例的回调字典中添加集成回调函数。

    Add integration callbacks to the instance's callbacks dictionary.

    该函数根据实例类型加载并添加各种第三方集成的回调函数。所有实例都会接收 HUB 和平台回调，
    而训练器实例还会额外接收 ClearML、Comet、DVC、MLflow、Neptune、Ray Tune、TensorBoard
    和 Weights & Biases 等集成工具的回调函数。这种设计允许 Ultralytics 与多种实验跟踪和
    可视化工具无缝集成。

    Args:
        instance (Trainer | Predictor | Validator | Exporter): 要添加回调的对象实例。
            实例的类型决定了加载哪些回调函数。训练器实例会加载所有集成回调，
            其他实例只加载基础的 HUB 和平台回调。

    Examples:
        >>> from ultralytics.engine.trainer import BaseTrainer
        >>> trainer = BaseTrainer()
        >>> add_integration_callbacks(trainer)
    """
    from .hub import callbacks as hub_cb  # HUB 集成回调
    from .platform import callbacks as platform_cb  # 平台集成回调

    # Load Ultralytics callbacks
    # 加载 Ultralytics 基础回调（所有实例都需要）
    callbacks_list = [hub_cb, platform_cb]

    # Load training callbacks
    # 加载训练专用回调（仅训练器实例需要）
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb  # ClearML 实验跟踪
        from .comet import callbacks as comet_cb  # Comet ML 实验跟踪
        from .dvc import callbacks as dvc_cb  # DVC 数据版本控制
        from .mlflow import callbacks as mlflow_cb  # MLflow 实验跟踪
        from .neptune import callbacks as neptune_cb  # Neptune AI 实验跟踪
        from .raytune import callbacks as tune_cb  # Ray Tune 超参数调优
        from .tensorboard import callbacks as tb_cb  # TensorBoard 可视化
        from .wb import callbacks as wb_cb  # Weights & Biases 实验跟踪

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # Add the callbacks to the callbacks dictionary
    # 将回调函数添加到实例的回调字典中（避免重复添加）
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:  # 检查回调是否已存在，避免重复
                instance.callbacks[k].append(v)
