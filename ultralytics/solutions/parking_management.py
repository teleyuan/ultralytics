from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imshow


class ParkingPtsSelection:
    """
    停车位点选择(ParkingPtsSelection)类：使用基于Tkinter的UI在图像上选择和管理停车区域点

    该类提供上传图像、选择点以定义停车区域以及将选定的点保存到JSON文件的功能。
    使用Tkinter构建图形用户界面，支持交互式地标注停车位边界框。

    核心功能：
    1. 上传并显示图像
    2. 通过鼠标点击选择停车位的四个角点
    3. 可视化显示已选择的停车位边界框
    4. 支持删除最后一个边界框
    5. 将标注结果保存为JSON格式

    属性:
        tk (module): Tkinter模块，用于GUI操作
        filedialog (module): Tkinter的文件对话框模块，用于文件选择操作
        messagebox (module): Tkinter的消息框模块，用于显示消息框
        master (tk.Tk): 主Tkinter窗口
        canvas (tk.Canvas): 用于显示图像和绘制边界框的画布控件
        image (PIL.Image.Image): 上传的图像
        canvas_image (ImageTk.PhotoImage): 在画布上显示的图像
        rg_data (list[list[tuple[int, int]]]): 边界框列表，每个边界框由4个点定义
        current_box (list[tuple[int, int]]): 当前边界框点的临时存储
        imgw (int): 上传图像的原始宽度
        imgh (int): 上传图像的原始高度
        canvas_max_width (int): 画布的最大宽度
        canvas_max_height (int): 画布的最大高度

    方法:
        initialize_properties: 初始化图像、画布、边界框和尺寸的属性
        upload_image: 上传并在画布上显示图像，调整大小以适应指定尺寸
        on_canvas_click: 处理鼠标点击以在画布上添加边界框点
        draw_box: 使用提供的坐标在画布上绘制边界框
        remove_last_bounding_box: 从列表中删除最后一个边界框并重绘画布
        redraw_canvas: 重绘画布，包括图像和所有边界框
        save_to_json: 将选定的停车区域点保存到JSON文件，并进行坐标缩放

    使用示例:
        >>> parking_selector = ParkingPtsSelection()
        >>> # 使用GUI上传图像，选择停车区域，并保存数据
    """

    def __init__(self) -> None:
        """
        初始化ParkingPtsSelection类，设置UI和用于停车区域点选择的属性

        该方法会创建一个Tkinter窗口，提供图像上传、点选择和保存功能。
        """
        try:  # 检查是否安装了tkinter
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:  # 显示错误和建议
            import platform

            install_cmd = {
                "Linux": "sudo apt install python3-tk (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | "
                "sudo pacman -S tk (Arch)",
                "Windows": "重新安装Python并在安装过程中启用 **可选功能** 中的 `tcl/tk and IDLE` 复选框",
                "Darwin": "从 https://www.python.org/downloads/macos/ 重新安装Python 或使用 `brew install python-tk`",
            }.get(platform.system(), "未知操作系统。请检查您的Python安装。")

            LOGGER.warning(f" Tkinter未配置或不受支持。可能的修复方法: {install_cmd}")
            return

        if not check_imshow(warn=True):
            return

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()  # 主应用程序窗口的引用
        self.master.title("Ultralytics 停车区域点选择器")
        self.master.resizable(False, False)

        self.canvas = self.tk.Canvas(self.master, bg="white")  # 用于显示图像的画布控件
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None  # 存储加载图像的变量
        self.canvas_image = None  # 画布上显示的图像的引用
        self.canvas_max_width = None  # 画布的最大允许宽度
        self.canvas_max_height = None  # 画布的最大允许高度
        self.rg_data = None  # 区域标注管理数据
        self.current_box = None  # 存储当前选择的边界框
        self.imgh = None  # 当前图像的高度
        self.imgw = None  # 当前图像的宽度

        # 带按钮的按钮框架
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("上传图像", self.upload_image),
            ("删除最后一个边界框", self.remove_last_bounding_box),
            ("保存", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self) -> None:
        """初始化图像、画布、边界框和尺寸的属性"""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self) -> None:
        """上传并在画布上显示图像，调整大小以适应指定尺寸"""
        from PIL import Image, ImageTk  # 作用域导入，因为ImageTk需要tkinter包

        file = self.filedialog.askopenfilename(filetypes=[("图像文件", "*.png *.jpg *.jpeg")])
        if not file:
            LOGGER.info("未选择图像。")
            return

        self.image = Image.open(file)
        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event) -> None:
        """处理鼠标点击以在画布上添加边界框点"""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box: list[tuple[int, int]]) -> None:
        """使用提供的坐标在画布上绘制边界框"""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self) -> None:
        """从列表中删除最后一个边界框并重绘画布"""
        if not self.rg_data:
            self.messagebox.showwarning("警告", "没有可删除的边界框。")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        """重绘画布，包括图像和所有边界框"""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self) -> None:
        """将选定的停车区域点保存到JSON文件，并进行坐标缩放"""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        from io import StringIO  # 函数级导入，因为仅在存储坐标时需要

        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())
        self.messagebox.showinfo("成功", "边界框已保存到 bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """
    停车管理(ParkingManagement)类：使用YOLO模型进行实时监控和可视化停车位占用和可用情况

    该类继承自BaseSolution，提供停车场管理功能，包括检测已占用车位、可视化停车区域以及显示占用统计信息。
    通过读取包含停车位坐标的JSON文件，对每个停车位进行占用检测，并实时更新统计数据。

    核心功能：
    1. 加载停车位区域坐标（JSON格式）
    2. 检测车辆并判断停车位占用状态
    3. 使用不同颜色区分已占用和空闲车位
    4. 实时显示占用和可用车位统计

    属性:
        json_file (str): 包含停车区域详细信息的JSON文件路径
        json (list[dict]): 加载的包含停车区域信息的JSON数据
        pr_info (dict[str, int]): 存储停车信息的字典（已占用和可用车位）
        arc (tuple[int, int, int]): 可用区域可视化的BGR颜色元组
        occ (tuple[int, int, int]): 已占用区域可视化的BGR颜色元组
        dc (tuple[int, int, int]): 检测到目标质心可视化的BGR颜色元组

    方法:
        process: 处理输入图像以进行停车场管理和可视化

    使用示例:
        >>> from ultralytics.solutions import ParkingManagement
        >>> parking_manager = ParkingManagement(model="yolo11n.pt", json_file="parking_regions.json")
        >>> print(f"已占用车位: {parking_manager.pr_info['Occupancy']}")
        >>> print(f"可用车位: {parking_manager.pr_info['Available']}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化停车管理系统，配置YOLO模型和可视化设置

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - model: YOLO模型路径
                - json_file: 停车区域坐标JSON文件路径
        """
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # 加载停车区域JSON数据
        if not self.json_file:
            LOGGER.warning("ParkingManagement需要包含停车区域坐标的 `json_file`。")
            raise ValueError("❌ JSON文件路径不能为空。")

        with open(self.json_file, encoding="utf-8") as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # 停车信息字典

        self.arc = (0, 0, 255)  # 可用区域颜色（红色）
        self.occ = (0, 255, 0)  # 已占用区域颜色（绿色）
        self.dc = (255, 0, 189)  # 每个框质心的颜色

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        处理输入图像以进行停车场管理和可视化

        该方法实现完整的停车场管理流程：
        1. 提取追踪目标
        2. 遍历JSON中定义的每个停车位区域
        3. 对于每个检测到的目标：
           - 计算边界框质心坐标
           - 使用pointPolygonTest判断质心是否在停车位多边形内
           - 如果在内部，标记该停车位为已占用
        4. 统计已占用和可用车位数量
        5. 使用不同颜色绘制停车位边界：
           - 已占用：绿色
           - 空闲：红色
        6. 在图像上显示统计信息

        Args:
            im0 (np.ndarray): 输入的推理图像

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - filled_slots: 已占用的停车位数量
                - available_slots: 可用的停车位数量
                - total_tracks: 追踪的目标总数

        使用示例:
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> results = parking_manager.process(image)
        """
        self.extract_tracks(im0)  # 从im0提取追踪轨迹
        available_slots, occupied_slots = len(self.json), 0
        annotator = SolutionAnnotator(im0, self.line_width)  # 初始化标注器

        for region in self.json:
            # 将点转换为具有正确dtype的NumPy数组并正确地reshape
            region_polygon = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            region_occupied = False
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                inside_distance = cv2.pointPolygonTest(region_polygon, (xc, yc), False)
                if inside_distance >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    region_occupied = True
                    break
            if region_occupied:
                occupied_slots += 1
                available_slots -= 1
            # 绘制区域
            cv2.polylines(
                im0, [region_polygon], isClosed=True, color=self.occ if region_occupied else self.arc, thickness=2
            )

        self.pr_info["Occupancy"], self.pr_info["Available"] = occupied_slots, available_slots

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=self.pr_info["Occupancy"],
            available_slots=self.pr_info["Available"],
            total_tracks=len(self.track_ids),
        )
