import time
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\mtyolov8sss_myCBAM.yaml',verbose=True)  # load a pretrained model (recommended for training)
    results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
    time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用