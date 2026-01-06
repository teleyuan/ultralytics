"""
快速训练脚本（使用内置 COCO8 数据集）
功能：使用 YOLO 自带的 coco8 数据集快速测试训练流程
"""

from ultralytics import YOLO
import torch
import os

# ============================================================
# 训练配置参数
# ============================================================

model_path = "yolov8s.pt"
data = "ultralytics/cfg/datasets/coco8.yaml"

# 训练参数
epochs = 2              # 训练轮数
imgsz = 640             # 图像尺寸
batch = 16              # 批量大小
workers = 8             # 数据加载线程数

# 保存设置
project = "runs/train"      # 保存目录
name = "train_coco8"        # 实验名称
exist_ok = False            # 覆盖还是递增实验目录，False 则递增
save = True                 # 保存检查点
save_period = -1            # 仅保存最后和最佳模型

# 训练策略
patience = 50           # EarlyStopping 耐心值
pretrained = True       # 使用预训练权重

# 优化器设置
optimizer = "auto"      # 自动选择优化器
lr0 = 0.01              # 初始学习率
lrf = 0.01              # 最终学习率
momentum = 0.937        # SGD 动量
weight_decay = 0.0005   # 权重衰减
warmup_epochs = 3.0     # 预热轮数

# 损失权重
box = 7.5               # 边界框损失权重
cls = 0.5               # 分类损失权重
dfl = 1.5               # DFL 损失权重

# 数据增强
hsv_h = 0.015           # HSV-Hue 增强
hsv_s = 0.7             # HSV-Saturation 增强
hsv_v = 0.4             # HSV-Value 增强
degrees = 0.0           # 旋转
translate = 0.1         # 平移
scale = 0.5             # 缩放
fliplr = 0.5            # 左右翻转
mosaic = 1.0            # Mosaic 增强

# 其他设置
verbose = True          # 详细输出
amp = True              # 自动混合精度（加速训练）
plots = True            # 生成训练图表

# ============================================================

def auto_select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 块 GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 自动使用第一块 GPU
        device = 0
        print(f"\n将使用 GPU 0 进行训练")
        return device
    else:
        print("未检测到 GPU")
        print("将使用 CPU 进行训练")
        return 'cpu'

def main():
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(data):
        print(f"\n错误：数据集配置文件 '{data}' 不存在！")
        return

    device = auto_select_device()

    # 加载模型
    model = YOLO(model_path, task="detect", verbose=False)
    model.to(device)

    try:
        results = model.train(
            # 数据配置
            data = data,

            # 训练参数
            epochs = epochs,
            imgsz = imgsz,
            batch = batch,
            device = device,
            workers = workers,

            # 保存设置
            project = project,
            name = name,
            exist_ok = exist_ok,
            save = save,
            save_period = save_period,

            # 训练策略
            patience = patience,
            pretrained = pretrained,

            # 优化器设置
            optimizer = optimizer,
            lr0 = lr0,
            lrf = lrf,
            momentum = momentum,
            weight_decay = weight_decay,
            warmup_epochs = warmup_epochs,

            # 损失权重
            box = box,
            cls = cls,
            dfl = dfl,

            # 数据增强
            hsv_h = hsv_h,
            hsv_s = hsv_s,
            hsv_v = hsv_v,
            degrees = degrees,
            translate = translate,
            scale = scale,
            fliplr = fliplr,
            mosaic = mosaic,

            # 其他设置
            verbose = verbose,
            amp = amp,
            plots = plots,
        )

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)

        # 获取训练结果目录
        save_dir = results.save_dir if hasattr(results, 'save_dir') else "runs/train/train_coco8"

        # 验证最佳模型
        print("\n正在验证最佳模型...")
        best_model_path = os.path.join(save_dir, "weights/best.pt")

        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data="coco8.yaml")

            print("\n最佳模型性能指标:")
            print("-" * 60)
            print(f"mAP50-95:  {metrics.box.map:.4f}   (主要指标)")
            print(f"mAP50:     {metrics.box.map50:.4f}  (IoU=0.5 时的 mAP)")
            print(f"mAP75:     {metrics.box.map75:.4f}  (IoU=0.75 时的 mAP)")
            print(f"Precision: {metrics.box.mp:.4f}   (精确率)")
            print(f"Recall:    {metrics.box.mr:.4f}   (召回率)")
            print("-" * 60)
        else:
            print(f"\n警告：未找到最佳模型文件")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
