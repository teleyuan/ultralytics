"""
图像检测脚本
功能：批量处理图像，进行目标检测并保存结果
"""

from ultralytics import YOLO
import os
import cv2

def main():
    """主函数"""
    # 加载模型（可以根据需要选择不同的模型）
    print("正在加载模型...")
    print("提示：如果是第一次运行，模型会自动从网上下载，请稍等...\n")

    model = YOLO("yolov8s.pt")  # YOLOv8 small 版本
    # model = YOLO("yolov8n.pt")  # YOLOv8 nano 版本（最快）
    # model = YOLO("yolov8m.pt")  # YOLOv8 medium 版本
    # model = YOLO("yolov8l.pt")  # YOLOv8 large 版本

    print(f"\n模型加载完成！类别数: {len(model.names)}")

    # 配置路径
    image_dir = "ultralytics/assets"        # 输入图像文件夹
    output_dir = "outputs"      # 输出结果文件夹

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查输入目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误：图像文件夹 '{image_dir}' 不存在！")
        print(f"请创建该文件夹并放入图像文件")
        return

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    if not image_files:
        print(f"错误：'{image_dir}' 文件夹中没有找到图像文件！")
        return

    print(f"\n找到 {len(image_files)} 张图像，开始处理...\n")

    # 批量处理图像
    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_name)

        print(f"[{idx}/{len(image_files)}] 处理: {img_name}")

        # 推理
        results = model(
            img_path,
            conf=0.25,      # 置信度阈值
            iou=0.45,       # NMS IoU 阈值
            verbose=False   # 不打印详细信息
        )
        result = results[0]

        # 获取检测结果
        boxes = result.boxes
        detections = []

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

                print(f"  - {class_name}: {conf:.2%} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print(f"  - 未检测到任何目标")

        # 保存结果图像
        output_path = os.path.join(output_dir, img_name)
        result.save(output_path)
        print(f"  ✓ 结果已保存到: {output_path}\n")

    print(f"完成！所有结果已保存到 '{output_dir}' 文件夹")


if __name__ == "__main__":
    main()
