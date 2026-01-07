"""
图像检测脚本
功能：批量处理图像，进行目标检测并保存结果
"""

from ultralytics import YOLO
import os
import cv2
import torch

# ============================================================
# 预测配置参数
# ============================================================

# 模型和路径配置
model_path = "yolov8s.pt"              
image_dir = "../datasets/coco8/images/val"       
output_dir = "outputs/coco8"                 

# 预测参数
conf = 0.25             # 置信度阈值
iou = 0.45              # NMS IoU 阈值
max_det = 300           # 每张图像最大检测数
imgsz = 640             # 图像尺寸
verbose = False         # 是否打印详细信息

# 其他设置
show_info = True        # 是否显示模型信息

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
        print(f"\n将使用 GPU 0 进行推理")
        return device
    else:
        print("未检测到 GPU")
        print("将使用 CPU 进行推理")
        return 'cpu'

def main():
    if not os.path.exists(model_path):
        print(f"\n错误：模型文件 '{model_path}' 不存在！")
        return

    if not os.path.exists(image_dir):
        print(f"\n错误：图像文件夹 '{image_dir}' 不存在！")
        print(f"请创建该文件夹并放入图像文件")
        return

    print("正在加载模型...")

    device = auto_select_device()
    model = YOLO(model_path)
    #print(model.names)
    print(model.info())
    model.to(device)
    model.eval()

    if show_info:
        model.info(True, True)

    print(f"\n模型加载完成！类别数: {len(model.names)}")     

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    if not image_files:
        print(f"\n错误：'{image_dir}' 文件夹中没有找到图像文件！")
        return

    print(f"\n找到 {len(image_files)} 张图像，开始处理...\n")

    # 批量处理图像
    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_name)

        print(f"[{idx}/{len(image_files)}] 处理: {img_name}")

        # 推理
        results = model.predict(
            img_path,
            conf = conf,
            iou = iou,
            max_det = max_det,
            imgsz = imgsz,
            verbose = verbose,
            device = device
        )
        result = results[0]

        # 获取检测结果
        boxes = result.boxes
        detections = []

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

                print(f"  - {class_name}: {confidence:.2%} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print(f"  - 未检测到任何目标")

        # 保存结果图像
        output_path = os.path.join(output_dir, img_name)
        result.save(output_path)
        print(f"  结果已保存到: {output_path}\n")

    print(f"完成！所有结果已保存到 '{output_dir}' 文件夹")


if __name__ == "__main__":
    main()
