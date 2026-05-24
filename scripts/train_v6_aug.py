from ultralytics import YOLO

# v7: 使用 v6 清洗后的数据，但回退 v6 中偏强的正则化。
# v6 的标注方向是对的：mature 标准更统一，bbox 更贴合果实本体。
# 但 "更严格的数据 + dropout + 更高 weight_decay" 让模型偏保守，Recall 下降。
# 本轮只保留 v6 数据，训练参数回到更接近 v5 的学习能力。
model = YOLO("yolo11s.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v6/data.yaml",
    epochs=200,
    patience=40,
    imgsz=960,
    batch=16,
    device=0,
    seed=0,
    deterministic=True,

    # 学习率与损失权重
    cos_lr=True,
    cls=0.7,
    weight_decay=0.0005,  # 回退到 v5，避免小数据集下正则化过强
    dropout=0.0,          # v7 先不加 dropout，重点看 Recall 能否恢复

    # 数据增强：沿用 v5/v6 的有效配置
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
    name="train_v6_aug",
    exist_ok=False,
    save=True,
    plots=True,
)
