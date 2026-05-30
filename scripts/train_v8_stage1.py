"""
v8 stage1：单类 YOLO 检测器。只检测"果子"一个类，不区分成熟度。

目的：把 v6_aug 那个"既要定位又要细粒度分类"的任务，拆成纯定位。
预期 mAP50 会显著上涨（因为类间混淆不再扣分），变成一个干净的定位 baseline。
成熟度判定交给 stage2 的分类器。

数据：data/labeled_v6_single/data.yaml（由 prepare_v8_data.py 生成）
其他训练参数和 v6_aug 完全一致，保证可比。
"""
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v6_single/data.yaml",
    epochs=200,
    patience=40,
    imgsz=960,
    batch=16,
    device=0,
    seed=0,
    deterministic=True,

    cos_lr=True,
    cls=0.7,
    weight_decay=0.0005,
    dropout=0.0,

    mosaic=1.0,
    mixup=0.15,
    close_mosaic=20,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    project="/home/jerico/projects/fruit_detect/models",
    name="train_v8_stage1",
    exist_ok=False,
    save=True,
    plots=True,
)
