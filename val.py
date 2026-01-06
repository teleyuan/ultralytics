"""
模型性能测试脚本
功能：测试模型的准确率、召回率、mAP 等指标
"""

from ultralytics import YOLO
import os
import torch

# ============================================================
# 验证配置参数
# ============================================================

# 模型和数据配置
model_path = "yolov8s.pt"                             
data = "ultralytics/cfg/datasets/coco8.yaml"            

# 验证参数
split = 'val'           # 数据集划分: 'val', 'test', 'train'
imgsz = 640             # 图像尺寸
batch = 16              # 批量大小
conf = 0.001            # 置信度阈值（用于计算指标）
iou = 0.6               # NMS IoU 阈值
max_det = 300           # 每张图像最大检测数
workers = 8             # 数据加载线程数

# 保存设置
save_json = False       # 保存为 COCO JSON 格式
save_hybrid = False     # 保存混合标签
verbose = True          # 打印详细信息
plots = True            # 保存图表
project = "runs/val"    # 保存目录
name = "exp"            # 实验名称

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
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(data):
        print(f"\n错误：数据集配置文件 '{data}' 不存在！")
        return

    device = auto_select_device()
    model = YOLO(model_path)
    model.to(device)

    # 运行验证
    print("\n开始验证...\n")
    print("-" * 60)

    metrics = model.val(
        data = data,
        split = split,
        imgsz = imgsz,
        batch = batch,
        conf = conf,
        iou = iou,
        max_det = max_det,
        device = device,
        workers = workers,
        save_json = save_json,
        save_hybrid = save_hybrid,
        verbose = verbose,
        plots = plots,
        project = project,
        name = name,
    )

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

    # 打印详细指标
    print("\n检测指标 (Detection Metrics):")
    print("-" * 60)

    # mAP 指标
    print("\n1. mAP (Mean Average Precision) - 平均精度均值:")
    print(f"   mAP50-95:  {metrics.box.map:.4f}   主要指标（COCO 标准）")
    print(f"   mAP50:     {metrics.box.map50:.4f}  (IoU=0.5 时的 mAP)")
    print(f"   mAP75:     {metrics.box.map75:.4f}  (IoU=0.75 时的 mAP)")

    # Precision 和 Recall
    print("\n2. Precision (精确率) 和 Recall (召回率):")
    print(f"   Precision: {metrics.box.mp:.4f}   (预测为正的样本中真正为正的比例)")
    print(f"   Recall:    {metrics.box.mr:.4f}   (所有正样本中被正确预测的比例)")

    # F1 Score
    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
        print(f"   F1-Score:  {f1:.4f}   (Precision 和 Recall 的调和平均)")

    # 各类别 mAP
    print("\n3. 各类别 AP (Average Precision):")
    if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap'):
        for idx, ap_value in zip(metrics.box.ap_class_index, metrics.box.ap):
            class_name = model.names[int(idx)]
            print(f"   {class_name:15s}: {ap_value:.4f}")

    print("\n4. 可视化结果:")
    print("   可视化结果已保存到: runs/val/exp/")


if __name__ == "__main__":
    main()
