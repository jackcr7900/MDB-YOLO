import time
from ultralytics import YOLO


# if __name__ == '__main__':
#     model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\yolov8s.yaml',verbose=True).load('yolov8s.pt') # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=16, cache=True)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# if __name__ == '__main__':
#     model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\yolov8m.yaml',verbose=True).load('yolov8m.pt') # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=8, cache=True)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# if __name__ == '__main__':
#     model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\yolov8.yaml',verbose=True).load('yolov8n.pt') # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# if __name__ == '__main__':
#     model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\yolov8-EfficientViT_M0.yaml',verbose=True) # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# if __name__ == '__main__':
#     model = YOLO('D:\\Yolov\\ultralytics-8.2.0\\ultralytics\\cfg\\models\\v8\\yolov8-mobilenetv4.yaml',verbose=True) # load a pretrained model (recommended for training)
#     results = model.train(data='mydata.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
#     time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

if __name__ == '__main__':
    model = YOLO('/mnt/v88/ultralytics-8.2.0_test/ultralytics/cfg/models/v9/yolov9c.yaml',verbose=True) # load a pretrained model (recommended for training)
    results = model.train(data='mydata.yaml', epochs=200, imgsz=640, device=[0,], workers=16, batch=32, cache=False)  # 开始训练
    time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用