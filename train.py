from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='./Licence Plate datasets/data.yaml', epochs=100, imgsz=640, batch=16)

print("Training is completed")