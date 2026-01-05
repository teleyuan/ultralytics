from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class SecurityAlarm(BaseSolution):
    """
    安全报警(SecurityAlarm)类：用于实时监控的安全报警功能管理

    该类继承自BaseSolution类，提供监控帧中目标、在检测总数超过特定阈值时发送邮件通知、
    以及标注输出帧用于可视化的功能。主要应用于安全监控场景，当检测到异常数量的目标时自动发送报警邮件。

    核心功能：
    1. 实时监控视频帧中的目标数量
    2. 设定目标数量阈值触发报警
    3. 通过SMTP发送邮件报警（附带检测图像）
    4. 防止重复发送报警邮件

    属性:
        email_sent (bool): 标记当前事件是否已发送邮件的标志
        records (int): 触发报警的检测目标数量阈值
        server (smtplib.SMTP): 用于发送邮件报警的SMTP服务器连接
        to_email (str): 接收报警的收件人邮箱地址
        from_email (str): 发送报警的发件人邮箱地址

    方法:
        authenticate: 设置邮件服务器认证以发送报警
        send_email: 发送包含详细信息和图像附件的邮件通知
        process: 监控帧、处理检测并在超过阈值时触发报警

    使用示例:
        >>> from ultralytics.solutions import SecurityAlarm
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> results = security.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        初始化SecurityAlarm类，配置实时目标监控的参数

        Args:
            **kwargs (Any): 传递给父类的关键字参数，包括:
                - records: 触发报警的目标数量阈值
                - model: YOLO模型路径
        """
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email: str, password: str, to_email: str) -> None:
        """
        认证邮件服务器以发送报警通知

        该方法使用提供的凭据初始化与SMTP服务器的安全连接并登录。
        使用Gmail的SMTP服务器（smtp.gmail.com:587）建立TLS加密连接。

        Args:
            from_email (str): 发件人邮箱地址
            password (str): 发件人邮箱账户的密码或应用专用密码
            to_email (str): 收件人邮箱地址

        使用示例:
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com: 587")
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records: int = 5) -> None:
        """
        发送包含图像附件的邮件通知，指示检测到的目标数量

        该方法对输入图像进行编码，撰写包含检测详情的邮件消息，并将其发送给指定的收件人。
        邮件包含文本正文和JPEG格式的图像附件。

        Args:
            im0 (np.ndarray): 要附加到邮件的输入图像或帧
            records (int, optional): 要包含在邮件消息中的检测目标数量，默认为5

        使用示例:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # 将图像编码为JPEG格式

        # 创建邮件
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "安全报警"

        # 添加文本消息正文
        message_body = f"Ultralytics 报警: 检测到 {records} 个目标。"
        message.attach(MIMEText(message_body))

        # 附加图像
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        # 发送邮件
        try:
            self.server.send_message(message)
            LOGGER.info("邮件发送成功！")
        except Exception as e:
            LOGGER.error(f"邮件发送失败: {e}")

    def process(self, im0) -> SolutionResults:
        """
        监控帧、处理目标检测并在超过阈值时触发报警

        该方法实现完整的安全监控流程：
        1. 处理输入帧并提取检测结果
        2. 使用边界框标注帧
        3. 统计检测到的目标总数
        4. 如果目标数量超过指定阈值且尚未发送报警：
           - 发送包含当前帧的邮件通知
           - 标记报警已发送以防止重复发送
        5. 返回标注后的结果

        Args:
            im0 (np.ndarray): 待处理和标注的输入图像或帧

        Returns:
            (SolutionResults): 包含以下信息的结果对象：
                - plot_im: 处理后的图像
                - total_tracks: 追踪的目标总数
                - email_sent: 是否触发了邮件报警

        使用示例:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = alarm.process(frame)
            >>> if results.email_sent:
            ...     print("报警已触发！")
        """
        self.extract_tracks(im0)  # 提取追踪轨迹
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        # 遍历边界框和类别索引
        for box, cls in zip(self.boxes, self.clss):
            # 绘制边界框
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det >= self.records and not self.email_sent:  # 仅在之前未发送时发送邮件
            self.send_email(im0, total_det)
            self.email_sent = True

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回SolutionResults对象
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)
