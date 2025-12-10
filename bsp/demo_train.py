#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
from ultralytics import YOLO


# yolov8n模型训练：训练模型的数据为'A_my_data.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], workers=16, batch=32, cache=False)  # 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


# if __name__ == '__main__':
#     model = YOLO('/mnt/v88/ultralytics-8.2.0_test/ultralytics/cfg/models/v8/MDB-YOLOv8.yaml',verbose=True).load("yolov8s.pt")  # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], batch=32, workers=16, cache=False)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# YOLOv8s.yaml
# if __name__ == '__main__':
#     model = YOLO('/mnt/v88/ultralytics-8.2.0_test/ultralytics/cfg/models/v8/yolov8s.yaml',verbose=True).load("yolov8s.pt")  # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], batch=32, workers=16, cache=False)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
# # YOLOv8m.yaml
# if __name__ == '__main__':
#     model = YOLO('/mnt/v88/ultralytics-8.2.0_test/ultralytics/cfg/models/v8/yolov8m.yaml',verbose=True).load("yolov8m.pt")  # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], batch=16, workers=16, cache=False)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
# # YOLOv8l.yaml
# if __name__ == '__main__':
#     model = YOLO('/mnt/v88/ultralytics-8.2.0_test/ultralytics/cfg/models/v8/yolov8l.yaml',verbose=True).load("yolov8l.pt")  # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], batch=16, workers=16, cache=False)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用