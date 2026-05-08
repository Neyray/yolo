from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v4/data.yaml",
    epochs=150,
    imgsz=640,
    device=0,
    project="/home/jerico/projects/fruit_detect/models",
    name="train_v4",
    patience=30
)
