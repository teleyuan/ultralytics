# YOLO 调用链速查表

**精简版** - 仅包含文件路径、类名、函数名，无具体代码实现

---

## 目录

1. [训练调用链](#1-训练调用链-train)
2. [预测调用链](#2-预测调用链-predict)
3. [验证调用链](#3-验证调用链-validation)
4. [核心文件索引](#4-核心文件索引)

---

## 1. 训练调用链 (Train)

### 1.1 调用流程

```
① 用户脚本
   train.py
   └─ model.train(**kwargs)

② Model 类
   ultralytics/engine/model.py
   └─ Model.train()
       ├─ _smart_load("trainer")  # 加载训练器
       └─ trainer.train()

③ DetectionTrainer 类
   ultralytics/models/yolo/detect/train.py
   └─ DetectionTrainer.__init__()
       └─ super().__init__()  # 调用 BaseTrainer

④ BaseTrainer 类
   ultralytics/engine/trainer.py
   ├─ BaseTrainer.__init__()
   └─ BaseTrainer.train()
       └─ _do_train()  # 实际训练循环

⑤ 训练循环
   ultralytics/engine/trainer.py
   └─ BaseTrainer._do_train()
       ├─ _setup_train()        # 设置环境
       ├─ for epoch:
       │   ├─ for batch:
       │   │   ├─ preprocess_batch()    # 预处理
       │   │   ├─ model(batch)          # 前向传播
       │   │   ├─ loss.backward()       # 反向传播
       │   │   └─ optimizer_step()      # 更新权重
       │   ├─ validate()                # 验证
       │   └─ save_model()              # 保存模型
       └─ final_eval()                  # 最终评估
```

### 1.2 关键方法清单

| 步骤 | 文件 | 类 | 方法 |
|------|------|-----|------|
| 入口 | `train.py` | - | `main()` |
| 路由 | `engine/model.py` | `Model` | `train()` |
| 路由 | `engine/model.py` | `Model` | `_smart_load("trainer")` |
| 初始化 | `models/yolo/detect/train.py` | `DetectionTrainer` | `__init__()` |
| 初始化 | `engine/trainer.py` | `BaseTrainer` | `__init__()` |
| 启动 | `engine/trainer.py` | `BaseTrainer` | `train()` |
| 核心循环 | `engine/trainer.py` | `BaseTrainer` | `_do_train()` |
| 环境设置 | `engine/trainer.py` | `BaseTrainer` | `_setup_train()` |
| 模型加载 | `engine/trainer.py` | `BaseTrainer` | `setup_model()` |
| 优化器 | `engine/trainer.py` | `BaseTrainer` | `build_optimizer()` |
| 数据加载器 | `models/yolo/detect/train.py` | `DetectionTrainer` | `get_dataloader()` |
| 数据集构建 | `models/yolo/detect/train.py` | `DetectionTrainer` | `build_dataset()` |
| 学习率调度 | `engine/trainer.py` | `BaseTrainer` | `_setup_scheduler()` |
| 预处理 | `models/yolo/detect/train.py` | `DetectionTrainer` | `preprocess_batch()` |
| 前向传播 | `nn/tasks.py` | `DetectionModel` | `forward()` |
| 损失计算 | `nn/tasks.py` | `DetectionModel` | `loss()` |
| 反向传播 | `engine/trainer.py` | `BaseTrainer` | _(在 _do_train 中)_ |
| 权重更新 | `engine/trainer.py` | `BaseTrainer` | `optimizer_step()` |
| 验证 | `engine/trainer.py` | `BaseTrainer` | `validate()` |
| 保存模型 | `engine/trainer.py` | `BaseTrainer` | `save_model()` |
| 最终评估 | `engine/trainer.py` | `BaseTrainer` | `final_eval()` |

---

## 2. 预测调用链 (Predict)

### 2.1 调用流程

```
① 用户脚本
   detect.py / 任意脚本
   ├─ model(source, **kwargs)           # __call__ 方法
   └─ model.predict(source, **kwargs)   # 显式调用

② Model 类
   ultralytics/engine/model.py
   ├─ Model.__call__()                  # 语法糖
   │   └─ self.predict()
   └─ Model.predict()
       ├─ _smart_load("predictor")      # 加载预测器
       └─ predictor(source, stream)

③ DetectionPredictor 类
   ultralytics/models/yolo/detect/predict.py
   └─ DetectionPredictor.__init__()
       └─ super().__init__()            # 调用 BasePredictor

④ BasePredictor 类
   ultralytics/engine/predictor.py
   ├─ BasePredictor.__init__()
   └─ BasePredictor.__call__()
       └─ stream_inference()            # 推理循环

⑤ 推理循环
   ultralytics/engine/predictor.py
   └─ BasePredictor.stream_inference()
       └─ for batch in dataset:
           ├─ preprocess(batch)         # 预处理
           ├─ inference(batch)          # 推理
           ├─ postprocess(preds)        # NMS 后处理
           └─ Results(...)              # 封装结果
```

### 2.2 关键方法清单

| 步骤 | 文件 | 类 | 方法 |
|------|------|-----|------|
| 入口 | 用户脚本 | - | - |
| 语法糖 | `engine/model.py` | `Model` | `__call__()` |
| 路由 | `engine/model.py` | `Model` | `predict()` |
| 路由 | `engine/model.py` | `Model` | `_smart_load("predictor")` |
| 初始化 | `models/yolo/detect/predict.py` | `DetectionPredictor` | `__init__()` |
| 初始化 | `engine/predictor.py` | `BasePredictor` | `__init__()` |
| 启动 | `engine/predictor.py` | `BasePredictor` | `__call__()` |
| 数据源设置 | `engine/predictor.py` | `BasePredictor` | `setup_source()` |
| 推理循环 | `engine/predictor.py` | `BasePredictor` | `stream_inference()` |
| 预处理 | `engine/predictor.py` | `BasePredictor` | `preprocess()` |
| 单图预处理 | `engine/predictor.py` | `BasePredictor` | `preprocess_single()` |
| 推理 | `engine/predictor.py` | `BasePredictor` | `inference()` |
| 后处理 | `models/yolo/detect/predict.py` | `DetectionPredictor` | `postprocess()` |
| NMS | `utils/ops.py` | - | `non_max_suppression()` |
| 坐标缩放 | `utils/ops.py` | - | `scale_boxes()` |
| 结果封装 | `engine/results.py` | `Results` | `__init__()` |
| 保存结果 | `engine/predictor.py` | `BasePredictor` | `save_preds()` |

---

## 3. 验证调用链 (Validation)

### 3.1 调用流程

```
① 用户脚本 或 训练器
   val.py 或 trainer.validate()
   └─ model.val(**kwargs)

② Model 类
   ultralytics/engine/model.py
   └─ Model.val()
       ├─ _smart_load("validator")      # 加载验证器
       └─ validator(model=self.model)

③ DetectionValidator 类
   ultralytics/models/yolo/detect/val.py
   └─ DetectionValidator.__init__()
       └─ super().__init__()            # 调用 BaseValidator

④ BaseValidator 类
   ultralytics/engine/validator.py
   ├─ BaseValidator.__init__()
   └─ BaseValidator.__call__()          # 验证循环

⑤ 验证循环
   ultralytics/engine/validator.py
   └─ BaseValidator.__call__()
       ├─ init_metrics()                # 初始化指标
       ├─ for batch in dataloader:
       │   ├─ preprocess(batch)         # 预处理
       │   ├─ model(batch)              # 推理
       │   ├─ postprocess(preds)        # NMS 后处理
       │   ├─ update_metrics()          # 更新指标
       │   └─ plot_predictions()        # 可视化
       ├─ get_stats()                   # 获取统计
       ├─ finalize_metrics()            # 计算最终指标
       └─ print_results()               # 打印结果
```

### 3.2 关键方法清单

| 步骤 | 文件 | 类 | 方法 |
|------|------|-----|------|
| 入口 | `val.py` 或 `engine/trainer.py` | - | `validate()` |
| 路由 | `engine/model.py` | `Model` | `val()` |
| 路由 | `engine/model.py` | `Model` | `_smart_load("validator")` |
| 初始化 | `models/yolo/detect/val.py` | `DetectionValidator` | `__init__()` |
| 初始化 | `engine/validator.py` | `BaseValidator` | `__init__()` |
| 启动 | `engine/validator.py` | `BaseValidator` | `__call__()` |
| 指标初始化 | `engine/validator.py` | `BaseValidator` | `init_metrics()` |
| 数据加载器 | `engine/validator.py` | `BaseValidator` | `get_dataloader()` |
| 预处理 | `engine/validator.py` | `BaseValidator` | `preprocess()` |
| 推理 | _(在 __call__ 中)_ | - | `model(batch)` |
| 后处理 | `models/yolo/detect/val.py` | `DetectionValidator` | `postprocess()` |
| NMS | `utils/ops.py` | - | `non_max_suppression()` |
| 更新指标 | `models/yolo/detect/val.py` | `DetectionValidator` | `update_metrics()` |
| 批次准备 | `models/yolo/detect/val.py` | `DetectionValidator` | `_prepare_batch()` |
| 预测准备 | `models/yolo/detect/val.py` | `DetectionValidator` | `_prepare_pred()` |
| 批次处理 | `models/yolo/detect/val.py` | `DetectionValidator` | `_process_batch()` |
| 获取统计 | `engine/validator.py` | `BaseValidator` | `get_stats()` |
| 最终指标 | `engine/validator.py` | `BaseValidator` | `finalize_metrics()` |
| 指标计算 | `utils/metrics.py` | `DetMetrics` | `process()` |
| AP 计算 | `utils/metrics.py` | `DetMetrics` | `compute_ap()` |
| 打印结果 | `engine/validator.py` | `BaseValidator` | `print_results()` |
| 可视化 | `engine/validator.py` | `BaseValidator` | `plot_val_samples()` |
| 可视化 | `engine/validator.py` | `BaseValidator` | `plot_predictions()` |

---

## 4. 核心文件索引

### 4.1 文件结构速查

```
ultralytics/
├── __init__.py                     # 懒加载: __getattr__(), __dir__()
│
├── engine/
│   ├── model.py                    # Model 基类
│   │   ├─ train()                  # 训练入口
│   │   ├─ predict() / __call__()   # 预测入口
│   │   ├─ val()                    # 验证入口
│   │   └─ _smart_load(key)         # 动态加载组件
│   │
│   ├── trainer.py                  # BaseTrainer 基类
│   │   ├─ __init__()
│   │   ├─ train()                  # 训练启动
│   │   ├─ _do_train()              # 训练循环
│   │   ├─ _setup_train()           # 环境设置
│   │   ├─ _setup_scheduler()       # 学习率调度
│   │   ├─ setup_model()            # 模型加载
│   │   ├─ build_optimizer()        # 优化器构建
│   │   ├─ optimizer_step()         # 权重更新
│   │   ├─ validate()               # 验证
│   │   ├─ save_model()             # 保存模型
│   │   └─ final_eval()             # 最终评估
│   │
│   ├── predictor.py                # BasePredictor 基类
│   │   ├─ __init__()
│   │   ├─ __call__()               # 预测启动
│   │   ├─ setup_source()           # 数据源设置
│   │   ├─ stream_inference()       # 推理循环
│   │   ├─ preprocess()             # 预处理
│   │   ├─ preprocess_single()      # 单图预处理
│   │   ├─ inference()              # 推理
│   │   ├─ postprocess()            # 后处理（子类重写）
│   │   └─ save_preds()             # 保存结果
│   │
│   ├── validator.py                # BaseValidator 基类
│   │   ├─ __init__()
│   │   ├─ __call__()               # 验证启动
│   │   ├─ init_metrics()           # 指标初始化
│   │   ├─ get_dataloader()         # 数据加载器
│   │   ├─ preprocess()             # 预处理
│   │   ├─ postprocess()            # 后处理（子类重写）
│   │   ├─ update_metrics()         # 更新指标（子类重写）
│   │   ├─ get_stats()              # 获取统计
│   │   ├─ finalize_metrics()       # 最终指标
│   │   ├─ print_results()          # 打印结果
│   │   ├─ plot_val_samples()       # 可视化样本
│   │   └─ plot_predictions()       # 可视化预测
│   │
│   └── results.py                  # Results 类
│       └─ __init__()               # 结果封装
│
├── models/
│   └── yolo/
│       ├── model.py                # YOLO, YOLOWorld, YOLOE 类
│       │   ├─ YOLO.__init__()
│       │   └─ YOLO.task_map        # 任务路由表
│       │
│       └── detect/
│           ├── train.py            # DetectionTrainer
│           │   ├─ __init__()
│           │   ├─ build_dataset()
│           │   ├─ get_dataloader()
│           │   ├─ preprocess_batch()
│           │   ├─ set_model_attributes()
│           │   ├─ get_model()
│           │   └─ get_validator()
│           │
│           ├── predict.py          # DetectionPredictor
│           │   ├─ __init__()
│           │   └─ postprocess()    # NMS 后处理
│           │
│           └── val.py              # DetectionValidator
│               ├─ __init__()
│               ├─ postprocess()    # NMS 后处理
│               ├─ update_metrics()
│               ├─ _prepare_batch()
│               ├─ _prepare_pred()
│               └─ _process_batch()
│
├── nn/
│   └── tasks.py                    # 模型架构
│       ├─ DetectionModel
│       │   ├─ forward()            # 前向传播
│       │   ├─ loss()               # 损失计算
│       │   └─ predict()            # 推理预测
│       ├─ ClassificationModel
│       └─ SegmentationModel
│
├── data/
│   ├── build.py
│   │   ├─ build_dataloader()
│   │   └─ build_yolo_dataset()
│   └── augment.py                  # 数据增强
│
└── utils/
    ├── torch_utils.py
    │   ├─ ModelEMA                 # 指数移动平均
    │   ├─ EarlyStopping            # 早停
    │   ├─ select_device()          # 设备选择
    │   ├─ init_seeds()             # 随机种子
    │   └─ strip_optimizer()        # 精简优化器
    │
    ├── ops.py
    │   ├─ non_max_suppression()    # NMS
    │   ├─ scale_boxes()            # 坐标缩放
    │   └─ xywh2xyxy()              # 坐标转换
    │
    └── metrics.py
        ├─ DetMetrics               # 检测指标
        │   ├─ process()            # 计算 mAP, P, R
        │   └─ compute_ap()         # 计算 AP
        ├─ SegMetrics               # 分割指标
        └─ ClassifyMetrics          # 分类指标
```

### 4.2 类继承关系速查

```
Model (engine/model.py)
    ├─ train()
    ├─ predict() / __call__()
    ├─ val()
    └─ _smart_load(key)
        ↑
        └─ YOLO (models/yolo/model.py)
            └─ task_map (路由表)

BaseTrainer (engine/trainer.py)
    ├─ train()
    ├─ _do_train()
    └─ validate()
        ↑
        ├─ DetectionTrainer (models/yolo/detect/train.py)
        ├─ ClassificationTrainer (models/yolo/classify/train.py)
        └─ SegmentationTrainer (models/yolo/segment/train.py)

BasePredictor (engine/predictor.py)
    ├─ __call__()
    ├─ stream_inference()
    └─ postprocess()
        ↑
        ├─ DetectionPredictor (models/yolo/detect/predict.py)
        ├─ ClassificationPredictor (models/yolo/classify/predict.py)
        └─ SegmentationPredictor (models/yolo/segment/predict.py)

BaseValidator (engine/validator.py)
    ├─ __call__()
    ├─ update_metrics()
    └─ finalize_metrics()
        ↑
        ├─ DetectionValidator (models/yolo/detect/val.py)
        ├─ ClassificationValidator (models/yolo/classify/val.py)
        └─ SegmentationValidator (models/yolo/segment/val.py)
```

### 4.3 task_map 路由表

**文件**: `ultralytics/models/yolo/model.py`
**类**: `YOLO`
**属性**: `task_map`

```python
task_map = {
    "classify": {
        "model": ClassificationModel,
        "trainer": yolo.classify.ClassificationTrainer,
        "validator": yolo.classify.ClassificationValidator,
        "predictor": yolo.classify.ClassificationPredictor,
    },
    "detect": {
        "model": DetectionModel,
        "trainer": yolo.detect.DetectionTrainer,
        "validator": yolo.detect.DetectionValidator,
        "predictor": yolo.detect.DetectionPredictor,
    },
    "segment": {
        "model": SegmentationModel,
        "trainer": yolo.segment.SegmentationTrainer,
        "validator": yolo.segment.SegmentationValidator,
        "predictor": yolo.segment.SegmentationPredictor,
    },
    "pose": {...},
    "obb": {...},
}
```

---

## 5. 快速查找表

### 5.1 功能 → 入口方法

| 功能 | 入口方法 | 文件 | 类 |
|------|---------|------|-----|
| 训练 | `train()` | `engine/model.py` | `Model` |
| 预测 | `predict()` / `__call__()` | `engine/model.py` | `Model` |
| 验证 | `val()` | `engine/model.py` | `Model` |

### 5.2 任务 → 核心类

| 任务 | Trainer | Predictor | Validator |
|------|---------|-----------|-----------|
| 检测 (detect) | `DetectionTrainer` | `DetectionPredictor` | `DetectionValidator` |
| 分类 (classify) | `ClassificationTrainer` | `ClassificationPredictor` | `ClassificationValidator` |
| 分割 (segment) | `SegmentationTrainer` | `SegmentationPredictor` | `SegmentationValidator` |
| 姿态 (pose) | `PoseTrainer` | `PosePredictor` | `PoseValidator` |
| 旋转框 (obb) | `OBBTrainer` | `OBBPredictor` | `OBBValidator` |

### 5.3 操作 → 核心方法

| 操作 | 方法 | 文件 | 类 |
|------|------|------|-----|
| 训练循环 | `_do_train()` | `engine/trainer.py` | `BaseTrainer` |
| 推理循环 | `stream_inference()` | `engine/predictor.py` | `BasePredictor` |
| 验证循环 | `__call__()` | `engine/validator.py` | `BaseValidator` |
| 前向传播 | `forward()` | `nn/tasks.py` | `DetectionModel` |
| 损失计算 | `loss()` | `nn/tasks.py` | `DetectionModel` |
| NMS | `non_max_suppression()` | `utils/ops.py` | - |
| 指标计算 | `process()` | `utils/metrics.py` | `DetMetrics` |

### 5.4 训练关键步骤 → 方法

| 步骤 | 方法 | 文件 | 类 |
|------|------|------|-----|
| 1. 设置环境 | `_setup_train()` | `engine/trainer.py` | `BaseTrainer` |
| 2. 加载模型 | `setup_model()` | `engine/trainer.py` | `BaseTrainer` |
| 3. 构建优化器 | `build_optimizer()` | `engine/trainer.py` | `BaseTrainer` |
| 4. 数据加载器 | `get_dataloader()` | `models/yolo/detect/train.py` | `DetectionTrainer` |
| 5. 预处理 | `preprocess_batch()` | `models/yolo/detect/train.py` | `DetectionTrainer` |
| 6. 前向传播 | `forward()` | `nn/tasks.py` | `DetectionModel` |
| 7. 损失计算 | `loss()` | `nn/tasks.py` | `DetectionModel` |
| 8. 反向传播 | _(在 _do_train 中)_ | `engine/trainer.py` | `BaseTrainer` |
| 9. 更新权重 | `optimizer_step()` | `engine/trainer.py` | `BaseTrainer` |
| 10. 验证 | `validate()` | `engine/trainer.py` | `BaseTrainer` |
| 11. 保存模型 | `save_model()` | `engine/trainer.py` | `BaseTrainer` |

### 5.5 预测关键步骤 → 方法

| 步骤 | 方法 | 文件 | 类 |
|------|------|------|-----|
| 1. 设置数据源 | `setup_source()` | `engine/predictor.py` | `BasePredictor` |
| 2. 预处理 | `preprocess()` | `engine/predictor.py` | `BasePredictor` |
| 3. 推理 | `inference()` | `engine/predictor.py` | `BasePredictor` |
| 4. 后处理 (NMS) | `postprocess()` | `models/yolo/detect/predict.py` | `DetectionPredictor` |
| 5. 结果封装 | `__init__()` | `engine/results.py` | `Results` |
| 6. 保存结果 | `save_preds()` | `engine/predictor.py` | `BasePredictor` |

### 5.6 验证关键步骤 → 方法

| 步骤 | 方法 | 文件 | 类 |
|------|------|------|-----|
| 1. 初始化指标 | `init_metrics()` | `engine/validator.py` | `BaseValidator` |
| 2. 数据加载器 | `get_dataloader()` | `engine/validator.py` | `BaseValidator` |
| 3. 预处理 | `preprocess()` | `engine/validator.py` | `BaseValidator` |
| 4. 推理 | _(在 __call__ 中)_ | `engine/validator.py` | `BaseValidator` |
| 5. 后处理 (NMS) | `postprocess()` | `models/yolo/detect/val.py` | `DetectionValidator` |
| 6. 更新指标 | `update_metrics()` | `models/yolo/detect/val.py` | `DetectionValidator` |
| 7. 最终指标 | `finalize_metrics()` | `engine/validator.py` | `BaseValidator` |
| 8. 打印结果 | `print_results()` | `engine/validator.py` | `BaseValidator` |

---

## 6. 三大流程对比

| 维度 | 训练 (Train) | 预测 (Predict) | 验证 (Val) |
|------|-------------|---------------|-----------|
| **入口方法** | `Model.train()` | `Model()` / `Model.predict()` | `Model.val()` |
| **核心类** | `DetectionTrainer` | `DetectionPredictor` | `DetectionValidator` |
| **核心方法** | `BaseTrainer._do_train()` | `BasePredictor.stream_inference()` | `BaseValidator.__call__()` |
| **循环类型** | `for epoch → for batch` | `for batch in dataset` | `for batch in dataloader` |
| **模型状态** | `model.train()` | `model.eval()` | `model.eval()` |
| **是否更新权重** | ✅ 是 | ❌ 否 | ❌ 否 |
| **输出** | 模型 + 指标 | Results 对象 | Metrics 对象 |

---

**文档版本**: 1.0
**更新日期**: 2026-01-06
**适用版本**: Ultralytics YOLO 8.3.247
