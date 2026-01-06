"""
模型性能测试脚本
功能：测试模型的准确率、召回率、mAP 等指标
"""

from ultralytics import YOLO
import os

def main():
    """主函数"""
    print("=" * 60)
    print("YOLO 模型性能测试")
    print("=" * 60)

    # 配置
    model_path = "yolo11n.pt"           # 模型路径
    data_yaml = "data.yaml"             # 数据集配置文件

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        print("\n可用的模型选项:")
        print("  - yolo11n.pt (预训练模型)")
        print("  - runs/train/my_model/weights/best.pt (训练的最佳模型)")
        print("  - runs/train/my_model/weights/last.pt (训练的最后模型)")
        return

    # 检查数据集配置文件
    if not os.path.exists(data_yaml):
        print(f"\n错误：数据集配置文件 '{data_yaml}' 不存在！")
        return

    print(f"\n模型: {model_path}")
    print(f"数据集: {data_yaml}")

    # 加载模型
    print("\n正在加载模型...")
    model = YOLO(model_path)

    # 运行验证
    print("\n开始验证...\n")
    print("-" * 60)

    metrics = model.val(
        data=data_yaml,
        split='val',           # 数据集划分: 'val', 'test', 'train'
        imgsz=640,             # 图像尺寸
        batch=16,              # 批量大小
        conf=0.001,            # 置信度阈值（用于计算指标）
        iou=0.6,               # NMS IoU 阈值
        max_det=300,           # 每张图像最大检测数
        device=0,              # GPU 设备 (0, 1, 2... 或 'cpu')
        workers=8,             # 数据加载线程数
        save_json=False,       # 保存为 COCO JSON 格式
        save_hybrid=False,     # 保存混合标签
        verbose=True,          # 打印详细信息
        plots=True,            # 保存图表
        project="runs/val",    # 保存目录
        name="exp",            # 实验名称
    )

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

    # 打印详细指标
    print("\n📊 检测指标 (Detection Metrics):")
    print("-" * 60)

    # mAP 指标
    print("\n1. mAP (Mean Average Precision) - 平均精度均值:")
    print(f"   mAP50-95:  {metrics.box.map:.4f}   ⭐ 主要指标（COCO 标准）")
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

    # 混淆矩阵
    print("\n4. 混淆矩阵:")
    print("   混淆矩阵已保存到: runs/val/exp/confusion_matrix.png")

    # 其他结果
    print("\n5. 可视化结果:")
    print("   - 混淆矩阵: runs/val/exp/confusion_matrix.png")
    print("   - F1 曲线:  runs/val/exp/F1_curve.png")
    print("   - PR 曲线:  runs/val/exp/PR_curve.png")
    print("   - P 曲线:   runs/val/exp/P_curve.png")
    print("   - R 曲线:   runs/val/exp/R_curve.png")

    # 指标解释
    print("\n" + "=" * 60)
    print("📖 指标说明:")
    print("-" * 60)
    print("""
1. mAP50-95 (0-1，越大越好)
   - COCO 标准的主要指标
   - 在 IoU 阈值从 0.5 到 0.95（步长 0.05）的平均 mAP
   - 综合评估模型在不同 IoU 要求下的表现

2. mAP50 (0-1，越大越好)
   - IoU 阈值为 0.5 时的 mAP
   - 较宽松的指标，更容易达到高分

3. Precision 精确率 (0-1，越大越好)
   - 预测为正例的样本中，真正为正例的比例
   - 高 Precision 意味着误报少

4. Recall 召回率 (0-1，越大越好)
   - 所有真实正例中，被正确预测的比例
   - 高 Recall 意味着漏检少

5. F1-Score (0-1，越大越好)
   - Precision 和 Recall 的调和平均
   - 综合评估 Precision 和 Recall 的平衡

参考标准（针对通用目标检测）:
  - mAP50-95 > 0.5:  优秀
  - mAP50-95 > 0.3:  良好
  - mAP50-95 < 0.3:  需要改进
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
