from ultralytics import YOLO

model = YOLO("MDB-YOLO.pt")

output = model.export(format="onnx", simplify=True, dynamic=False, opset=16)
