from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 用预训练权重作为起点

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled/data.yaml",
    epochs=100,
    imgsz=640,
    device=0,  # 用GPU
    project="/home/jerico/projects/fruit_detect/models",
    name="train_v1"
)
