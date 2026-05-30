"""
v7：小目标方向。yolo11s + P2 检测头。

和 v6_aug 的唯一差别是模型结构：v6_aug 用标准 yolo11s（P3/P4/P5 三个检测头），
v7 换成 yolo11s-p2.yaml（增加一个 P2/4 检测头）。其余训练参数完全沿用 v6_aug，
保证 v6_aug → v7 是单变量对照，能直接看出 P2 头对小目标的收益。

数据集仍然是 data/labeled_v6，不动数据。
"""
from ultralytics import YOLO

# 用 P2 头的结构定义，但 backbone 用 yolo11s.pt 预训练权重初始化。
# .load() 只会拷贝匹配的层，检测头会随机初始化重新学。
model = YOLO("/home/jerico/projects/fruit_detect/configs/yolo11s-p2.yaml").load("yolo11s.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v6/data.yaml",
    epochs=200,
    patience=40,
    imgsz=960,
    batch=16,
    device=0,
    seed=0,
    deterministic=True,

    # 学习率与损失权重（沿用 v6_aug）
    cos_lr=True,
    cls=0.7,
    weight_decay=0.0005,
    dropout=0.0,

    # 数据增强（沿用 v6_aug）
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

    # 输出
    project="/home/jerico/projects/fruit_detect/models",
    name="train_v7",
    exist_ok=False,
    save=True,
    plots=True,
)
