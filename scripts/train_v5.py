from ultralytics import YOLO

# v5: 升级到 yolo11s，加强增强、加早停、加色彩抖动
# 目的：在新补充的 3 类（immature/mature/overripe）数据上，
# 让模型在 overripe 这种少样本类上有更稳的表现。
model = YOLO("yolo11s.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v5/data.yaml",
    epochs=200,
    patience=30,         # 早停：valid 30 epoch 没提升就停，避免过拟合
    imgsz=960,           # v4 数据 97% 是小目标(中位 box ≈38px)，640 不够；显存不够可降回 640，再不行加 batch=8
    batch=16,            # 配 imgsz=960，若 OOM 改 8
    device=0,
    cos_lr=True,         # 余弦退火学习率
    cls=0.7,             # 加大分类损失权重，缓解类不平衡(mature:immature:overripe = 24:9:1)
    # 几何增强
    mosaic=1.0,
    mixup=0.15,
    close_mosaic=20,     # 最后 20 epoch 关掉 mosaic，让模型在干净样本上收敛
    # 色彩增强：成熟度判别强依赖颜色，这里要给够变化
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    # 几何
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    # 输出
    project="/home/jerico/projects/fruit_detect/models",
    name="train_v5",
    exist_ok=False,
    save=True,
    plots=True,          # 训练完自动出 results.png / confusion_matrix.png / PR_curve.png
)
