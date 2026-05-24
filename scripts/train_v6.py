from ultralytics import YOLO

# v6: 在 v5 基础上做正则化实验，不更换数据集和模型规模。
# v5 的主要问题是 30~40 轮后验证集 loss 微升、Recall 偏低，
# 所以本轮先尝试提高 weight_decay 和加入 dropout，观察是否能缓解过拟合。
# 当前 Ultralytics 配置中没有 label_smoothing 参数，因此这里不写入，避免训练启动时报 unknown argument。
model = YOLO("yolo11s.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v6/data.yaml",
    epochs=200,
    patience=40,         # v5 best 在 30~40 轮附近，略放宽早停窗口观察正则化后是否还能涨
    imgsz=960,           # 保留 v5 的小目标设置；显存不足时先降 batch=8，再考虑 imgsz=768
    batch=16,
    device=0,
    seed=0,
    deterministic=True,

    # 学习率与损失权重
    cos_lr=True,
    cls=0.7,             # 保持 v5 分类损失权重，先隔离正则化变量
    weight_decay=0.001,  # v5 为 0.0005，本轮加大权重衰减抑制过拟合
    dropout=0.1,         # v6 正则化重点；若当前版本对 detect 忽略该项，args.yaml 中仍可确认记录

    # 数据增强：沿用 v5，避免一次改变太多变量
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
    name="train_v6",
    exist_ok=False,
    save=True,
    plots=True,
)
