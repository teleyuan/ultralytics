"""
Ultralytics YOLO 引擎模块初始化文件

该模块是 Ultralytics YOLO 引擎的核心组件集合，包含训练、验证、预测、导出等功能的基础类。
引擎模块提供了模型操作的统一接口和实现。

主要组件:
    - BaseTrainer: 训练器基类，处理模型训练流程
    - BaseValidator: 验证器基类，处理模型验证和评估
    - BasePredictor: 预测器基类，处理模型推理
    - Exporter: 导出器类，将模型导出为各种格式
    - Model: 模型基类，统一所有 YOLO 模型的接口
    - Results: 结果类，封装预测结果
    - Tuner: 超参数调优器
"""
