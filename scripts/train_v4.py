from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="/home/jerico/projects/fruit_detect/data/labeled_v4/data.yaml",
    epochs=150,
    imgsz=640,
    device=0,
    project="/home/jerico/projects/fruit_detect/models",
    name="train_v4_aug",      # 新名字，不覆盖上一次结果
    patience=0,                # 关闭 EarlyStopping，跑满 150 轮
    save=True,
    save_period=-1,            # 只保留 best.pt 和 last.pt
    plots=True,                # 自动生成 results.png 等曲线图

    # ---- 数据增强（小数据集场景增强） ----
    mosaic=1.0,                # 默认就开，4 图拼接
    close_mosaic=15,           # 最后 15 轮关闭 mosaic，让模型在干净图上收敛
    mixup=0.15,                # 两图混合，默认 0.0，开一点提升泛化
    copy_paste=0.1,            # 复制粘贴目标，默认 0.0，对小目标 + 类别不平衡有帮助
    hsv_h=0.015,               # 色调
    hsv_s=0.7,                 # 饱和度
    hsv_v=0.4,                 # 亮度（仰视场景天空亮度差异大，保留较高）
    degrees=10.0,              # 旋转 ±10°（默认 0）
    translate=0.1,             # 平移
    scale=0.5,                 # 缩放
    shear=2.0,                 # 错切（默认 0）
    perspective=0.0,           # 透视，仰视场景慎用，先关
    flipud=0.0,                # 上下翻转关闭（仰视果实有方向性）
    fliplr=0.5,                # 左右翻转保留
    erasing=0.4,               # 随机擦除
)
