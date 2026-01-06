"""
快速训练脚本（使用内置 COCO8 数据集）
功能：使用 YOLO 自带的 coco8 数据集快速测试训练流程
"""

from ultralytics import YOLO
import torch
import os

def auto_select_device():
    """
    自动选择训练设备（有 GPU 则使用 GPU，否则使用 CPU）
    """
    print("\n" + "=" * 60)
    print("设备检测")
    print("=" * 60)

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
    """主函数"""

    # 自动检测并选择设备
    device = auto_select_device()

    # 训练配置（使用默认值，适合快速测试）
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)

    # 默认参数（适合快速测试）
    model_name = "yolov8s.pt"  
    epochs = 2                # 训练 2 轮
    batch_size = 16            # 批量大小
    imgsz = 640                # 图像尺寸

    print(f"  模型: {model_name}")
    print(f"  数据集: coco8")
    print(f"  设备: {device}")
    print(f"  训练轮数: {epochs}")
    print(f"  批量大小: {batch_size}")
    print(f"  图像尺寸: {imgsz}")

    # 加载模型
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    model = YOLO(model_name)

    try:
        results = model.train(
            # 数据配置（使用内置 coco8 数据集）
            data="ultralytics/cfg/datasets/coco8.yaml",        # 内置的 COCO8 数据集

            # 训练参数
            epochs=epochs,            # 训练轮数
            imgsz=imgsz,              # 图像尺寸
            batch=batch_size,         # 批量大小
            device=device,            # GPU/CPU 设备
            workers=8,                # 数据加载线程数

            # 保存设置
            project="runs/train",     # 保存目录
            name="train_coco8",               # 实验名称
            exist_ok=False,           # 自动递增实验编号
            save=True,                # 保存检查点
            save_period=-1,           # 仅保存最后和最佳模型

            # 训练策略
            patience=50,              # EarlyStopping 耐心值
            pretrained=True,          # 使用预训练权重

            # 优化器设置
            optimizer="auto",         # 自动选择优化器
            lr0=0.01,                 # 初始学习率
            lrf=0.01,                 # 最终学习率
            momentum=0.937,           # SGD 动量
            weight_decay=0.0005,      # 权重衰减
            warmup_epochs=3.0,        # 预热轮数

            # 损失权重
            box=7.5,                  # 边界框损失权重
            cls=0.5,                  # 分类损失权重
            dfl=1.5,                  # DFL 损失权重

            # 数据增强
            hsv_h=0.015,              # HSV-Hue 增强
            hsv_s=0.7,                # HSV-Saturation 增强
            hsv_v=0.4,                # HSV-Value 增强
            degrees=0.0,              # 旋转
            translate=0.1,            # 平移
            scale=0.5,                # 缩放
            fliplr=0.5,               # 左右翻转
            mosaic=1.0,               # Mosaic 增强

            # 其他设置
            verbose=True,             # 详细输出
            amp=True,                 # 自动混合精度（加速训练）
            plots=True,               # 生成训练图表
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

            print(f"\n训练结果保存在:")
            print(f"  目录: {save_dir}")
            print(f"  最佳模型: {best_model_path}")
            print(f"  最后模型: {os.path.join(save_dir, 'weights/last.pt')}")

            print(f"\n查看训练曲线:")
            print(f"  方法1: 打开 {os.path.join(save_dir, 'results.png')}")
            print(f"  方法2: tensorboard --logdir runs/train")

            print(f"\n使用训练好的模型:")
            print(f"  from ultralytics import YOLO")
            print(f"  model = YOLO('{best_model_path}')")
            print(f"  results = model('image.jpg')")
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
