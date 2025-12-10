#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77
@File    ：start_window.py
@IDE     ：PyCharm
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：主要的图形化界面，本次图形化界面实现的主要技术为pyside6，pyside6是官方提供支持的
@Date    ：2024/8/15 15:15
'''
import copy                      # 用于图像复制
import os                        # 用于系统路径查找
import shutil                    # 用于复制
from PySide6.QtGui import *      # GUI组件
from PySide6.QtCore import *     # 字体、边距等系统变量
from PySide6.QtWidgets import *  # 窗口等小组件
import threading                 # 多线程
import sys                       # 系统库
import cv2                       # opencv图像处理
import torch                     # 深度学习框架
import os.path as osp            # 路径查找
import time                      # 时间计算
from ultralytics import YOLO     # yolo核心算法

# 常用的字符串常量
WINDOW_TITLE ="MDB-YOLO Based Residual Taro Peel Detection System"            # 系统上方标题
WELCOME_SENTENCE = "MDB-YOLO Based Residual Taro Peel Detection System Developed by GDUST AIOT Lab"   # 欢迎的句子
ICON_IMAGE = "images/UI/aaalogo2.png"                 # 系统logo界面
IMAGE_LEFT_INIT = "images/UI/up.jpeg"              # 图片检测界面初始化左侧图像
IMAGE_RIGHT_INIT = "images/UI/right.jpeg"          # 图片检测界面初始化右侧图像


#   一.主窗口初始化模块
class MainWindow(QTabWidget):
    def __init__(self):

        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)       # 系统界面标题
        self.resize(1200, 800)           # 系统初始化大小
        self.setWindowIcon(QIcon(ICON_IMAGE))   # 系统logo图像
        self.output_size = 480                  # 上传的图像和视频在系统界面上显示的大小
        self.img2predict = ""                   # 要进行预测的图像路径
        # self.device = 'cpu'

        # 视频相关参数
        self.init_vid_id = '0'  # 摄像头修改
        self.vid_source = int(self.init_vid_id)
        self.cap = cv2.VideoCapture(self.vid_source)
        self.stopEvent = threading.Event()


        self.webcam = True
        self.stopEvent.clear()

        # 模型和硬件初始化
        #self.model_path = "runs/detect/yolov8n/weights/best.pt"  # todo 指明模型加载的位置的设备
        self.model_path = "D:/Yolov/yolov8_taro_demo/yolov8-42/runs/detect/train2/weights/best.pt"
        self.model = self.model_load(weights=self.model_path)
        self.conf_thres = 0.25   # 置信度的阈值
        self.iou_thres = 0.45    # NMS操作的时候 IOU过滤的阈值

        # 视频相关参数
        self.vid_gap = 30        # 摄像头视频帧保存间隔。

        # 界面初始化
        self.initUI()            # 初始化图形化界面
        self.reset_vid()         # 重新设置视频参数，重新初始化是为了防止视频加载出错

    #  二.模型加载模块
    # 模型初始化 使用@torch.no_grad()装饰器禁用梯度计算，提升推理速度
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        模型加载
        """
        model_loaded = YOLO(weights)
        return model_loaded

    # 三.图形界面布局模块
    def initUI(self):
        """
        图形化界面初始化

        界面组成:
            1.图片检测页:
            左右分栏显示原图/检测结果
            上传图片和开始检测按钮
            使用QVBoxLayout和QHBoxLayout进行布局

            2.视频检测页:
                实时视频显示区域
                摄像头/视频文件检测按钮
                停止检测按钮
                多线程视频处理

            3.主页:
                欢迎界面
                模型切换功能
                历史记录查看
                作者信息
        """

        # ********************* 图片识别界面 *****************************
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("Image Detection")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        self.img_num_label = QLabel("Current Detection Result: Pending")
        self.img_num_label.setFont(font_main)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("Upload Image")
        det_img_button = QPushButton("Start Detection")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        # 八.样式设计
        """
        统一按钮样式
        使用CSS样式表进行控件美化
        响应觑标昙停效果
        统一的字体设置(楷体)
        """
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_num_label)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* 视频识别界面 *****************************
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("Video Detection")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("Webcam Real-time Detection")   #"Webcam Real-time Detection"
        self.mp4_detection_btn = QPushButton("Video File Detection")
        self.vid_stop_btn = QPushButton("Stop Detection")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        # todo 添加摄像头检测标签逻辑
        self.vid_num_label = QLabel("Current Detection Result: {}".format("Waiting"))
        self.vid_num_label.setFont(font_main)
        vid_detection_layout.addWidget(self.vid_num_label)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # ********************* 模型切换界面 *****************************
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_SENTENCE)
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/aaataro_1200x640.png'))
        self.model_label = QLabel("Current Model: {}".format(self.model_path))
        self.model_label.setFont(font_main)
        change_model_button = QPushButton("Switch Model")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")

        record_button = QPushButton("View History")
        record_button.setFont(font_main)
        record_button.clicked.connect(self.check_record)
        record_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href=''>Author: Feng Xingcan, fengxingcan@gdust.edu.cn</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        # 创建三个标签页
        self.addTab(about_widget, 'Home')
        self.addTab(img_detection_widget, 'Image Detection')
        self.addTab(vid_detection_widget, 'Video Detection')

        # 设置标签页图标
        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))

        # ********************* todo 布局修改和颜色变换等相关插件 *****************************

    # 四.图片检测模块
    """
    工作流程:
        1.通过QFileDialog选择图片文件
        2.将图片复制到临时目录并显示
        3.使用YOLO模型进行推理
        4.使用OpenCV处理检测结果并显示
        5.保存检测结果到record/img/目录
    """
    def upload_img(self):
        """上传图像，图像要尽可能保证是中文格式"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            # 判断用户是否选择了图像，如果用户选择了图像则执行下面的操作
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)  # 将图像转移到images目录下并且修改为英文的形式
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = save_path                               # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
            self.img_num_label.setText("Current Detection Result: Pending")

    # 六.模型切换模块
    """
    动态加载不同YOLOv8模型
    通过文件对话框选择,pt模型文件
    实时更新界面显示当前模型
    """
    def change_model(self):
        """切换模型，重新对self.model进行赋值"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            # 如果用户选择了对应的pt文件，根据用户选择的pt文件重新对模型进行初始化
            self.model_path = fileName
            self.model = self.model_load(weights=self.model_path)
            QMessageBox.information(self, "Success", "Model switched successfully!")
            self.model_label.setText("Current Model: {}".format(self.model_path))

    # 图片检测
    def detect_img(self):
        """检测单张的图像文件"""
        # txt_results = []
        output_size = self.output_size
        results = self.model(self.img2predict)  # 读取图像并执行检测的逻辑
        result = results[0]                     # 获取检测结果
        img_array = result.plot()               # 在图像上绘制检测结果
        im0 = img_array
        im_record = copy.deepcopy(im0)
        resize_scale = output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
        time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
        cv2.imwrite("record/img/{}.jpg".format(time_re), im_record)
        # 保存txt记录文件
        # if len(txt_results) > 0:
        #     np.savetxt('record/img/{}.txt'.format(time_re), np.array(txt_results), fmt="%s %s %s %s %s %s",
        #                delimiter="\n")
        # 获取预测出来的每个类别的数量并在对应的图形化检测页面上进行显示
        result_names = result.names
        result_nums = [0 for i in range(0, len(result_names))]
        cls_ids = list(result.boxes.cls.cpu().numpy())
        for cls_id in cls_ids:
            result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
        result_info = ""
        for idx_cls, cls_num in enumerate(result_nums):
            # 添加对数据0的判断，如果当前数据的数目为0，则这个数据不需要加入到里面
            if cls_num > 0:
                result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
        self.img_num_label.setText("Current Detection Result:\n {}".format(result_info))
        QMessageBox.information(self, "Detection Complete", "Log saved!")


    # 五.视频检测模块
    """
    工作流程:
        多线程处理视频流(防止界面卡顿)
        使用stopEvent进行线程控制
        支持摄像头和本地视频文件两种输入源
        定时保存检测结果到record/vid/目录
    """
    def open_cam(self):
        """打开摄像头上传"""
        self.webcam_detection_btn.setEnabled(False)    # 将打开摄像头的按钮设置为false，防止用户误触
        self.mp4_detection_btn.setEnabled(False)       # 将打开mp4文件的按钮设置为false，防止用户误触
        self.vid_stop_btn.setEnabled(True)             # 将关闭按钮打开，用户可以随时点击关闭按钮关闭实时的检测任务
        self.vid_source = int(self.init_vid_id)        # 重新初始化摄像头
        self.webcam = True                             # 将实时摄像头设置为true
        self.cap = cv2.VideoCapture(self.vid_source)   # 初始化摄像头的对象
        th = threading.Thread(target=self.detect_vid)  # 初始化视频检测线程  # 启动摄像头线程
        th.start()                                     # 启动线程进行检测

    def open_mp4(self):
        """打开mp4文件上传"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            # 和上面open_cam的方法类似，只是在open_cam的基础上将摄像头的源改为mp4的文件
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_source = fileName
            self.webcam = False
            self.cap = cv2.VideoCapture(self.vid_source)
            th = threading.Thread(target=self.detect_vid)
            th.start()

    # 视频检测主函数
    def detect_vid(self):
        """检测视频文件，这里的视频文件包含了mp4格式的视频文件和摄像头形式的视频文件"""
        # model = self.model
        vid_i = 0
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)
                result = results[0]
                img_array = result.plot()
                # 检测 展示然后保存对应的图像结果
                im0 = img_array
                im_record = copy.deepcopy(im0)
                resize_scale = self.output_size / im0.shape[0]
                im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
                if vid_i % self.vid_gap == 0:
                    cv2.imwrite("record/vid/{}.jpg".format(time_re), im_record)
                # 保存txt记录文件
                # if len(txt_results) > 0:
                #     np.savetxt('record/img/{}.txt'.format(time_re), np.array(txt_results), fmt="%s %s %s %s %s %s",
                #                delimiter="\n")
                # 统计每个类别的数目，如果这个类别检测到的数量大于0，则将这个类别在界面上进行展示
                result_names = result.names
                result_nums = [0 for i in range(0, len(result_names))]
                cls_ids = list(result.boxes.cls.cpu().numpy())
                for cls_id in cls_ids:
                    result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                result_info = ""
                for idx_cls, cls_num in enumerate(result_nums):
                    if cls_num > 0:
                        result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                    # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                    # result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
                self.vid_num_label.setText("Current Detection Result:\n {}".format(result_info))
                vid_i = vid_i + 1
            if cv2.waitKey(1) & self.stopEvent.is_set() == True:
                # 关闭并释放对应的视频资源
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                if self.cap is not None:
                    self.cap.release()
                    cv2.destroyAllWindows()
                self.reset_vid()
                break


    # 摄像头重置
    def reset_vid(self):
        """重置摄像头内容"""
        self.webcam_detection_btn.setEnabled(True)                      # 打开摄像头检测的按钮
        self.mp4_detection_btn.setEnabled(True)                         # 打开视频文件检测的按钮
        self.vid_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))                # 重新设置视频检测页面的初始化图像
        self.vid_source = int(self.init_vid_id)                         # 重新设置源视频源
        self.webcam = True                                              # 重新将摄像头设置为true
        self.vid_num_label.setText("Current Detection Result: {}".format("Waiting"))   # 重新设置视频检测页面的文字内容

    def close_vid(self):
        """关闭摄像头"""
        self.stopEvent.set()
        self.reset_vid()

    # 七.辅助功能模块
    """
    历史记录查看(图片/视频结果)
    安全的资源释放机制
    用户退出确认对话框
    """
    def check_record(self):
        """打开历史记录文件夹"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def closeEvent(self, event):
        """用户退出事件"""
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                # 退出之后一定要尝试释放摄像头资源，防止资源一直在线
                if self.cap is not None:
                    self.cap.release()
                    print("Camera released")
            except:
                pass
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())

""" 总结
九. 关键技术点总结
    1.多线程处理:视频检测使用独立线程，避免阻塞主界面
    2.0penCV集成:图像/视频的读取、处理和显示
    3.YOLOv8集成:直接调用Ultralytics官方API进行推理
    4.PySide6组件:
        QTabwidget实现多页签
        。QFileDialog进行文件选择
        。QLabel显示图片和文本
        信号槽机制实现事件处理
    5.结果记录系统:
        按时间戳保存检测结果
        分类存储图片和视频结果
        快速查看历史记录

十. 系统流程图
    启动程序
    - 初始化模型和界面
    - 图片检测流程:
    上传图片 →预处理 →推理 →显示结果 →保存记录
    - 视频检测流程:
    选择输入源 → 启动检测线程 →实时处理帧 →显示结果 → 保存记录
    该代码实现了一个完整的YOI08月标检测系统，须善图片,机版检测、模型切换，结果
"""

