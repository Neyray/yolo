# 训练实验记录

本文件记录 v4 / v4_aug / v5 三次训练的数据集、配置与结果。

类别：`immature_fruit` / `mature_fruit` / `overripe_fruit`（nc=3）

---

## 1. v4 训练（基线）

### 1.1 数据集

| 项目 | 数值 |
|---|---|
| 训练集图片 | 258 |
| 验证集图片 | 19 |
| 测试集图片 | 12 |
| mature_fruit 实例 | 3504 |
| immature_fruit 实例 | 1275 |
| overripe_fruit 实例 | 168 |

数据集路径：`data/labeled_v4/`

### 1.2 训练配置

| 参数 | 值 |
|---|---|
| model | yolo11n.pt |
| epochs | 150 |
| imgsz | 640 |
| batch | 16（自动） |
| device | 0 (GPU) |
| patience | 30 |
| optimizer | auto (AdamW) |

脚本：`scripts/train_v4.py`

### 1.3 训练结果

| 指标 | 数值 |
|---|---|
| 训练状态 | EarlyStopping 在第 57 轮触发，best 在第 27 轮 |
| mAP50 (all) | 0.239 |
| mAP50-95 (all) | 0.0785 |
| Precision | 0.401 |
| Recall | 0.256 |
| immature_fruit mAP50 | 0.254 |
| mature_fruit mAP50 | 0.381 |
| overripe_fruit mAP50 | 0.083 |

输出目录：`models/train_v4/`

---

## 2. v4_aug 训练（关闭早停 + 强化数据增强）

数据集与 v4 一致（`data/labeled_v4/`）。

### 2.1 训练配置变化

| 参数 | v4 | v4_aug |
|---|---|---|
| patience | 30 | 0 |
| mixup | 0.0 | 0.15 |
| copy_paste | 0.0 | 0.1 |
| degrees | 0 | 10 |
| shear | 0 | 2 |
| close_mosaic | 10 | 15 |
| flipud | 0 | 0 |
| fliplr | 0.5 | 0.5 |
| erasing | 0.4 | 0.4 |

脚本：`scripts/train_v4.py`（已更新为 v4_aug 配置）  
输出目录：`models/train_v4_aug/`

### 2.2 训练结果

| 指标 | 数值 |
|---|---|
| 训练状态 | 跑满 150 轮，best 在第 27 轮附近（mAP50 最高） |
| mAP50 (all) | 0.244 |
| mAP50-95 (all) | 0.084 |
| Precision | 0.401 |
| Recall | 0.256 |

### 2.3 训练曲线观察

| 曲线 | 走势 |
|---|---|
| train/box_loss | 2.8 → 1.3，持续下降 |
| train/cls_loss | 4.0 → 0.8，持续下降 |
| val/box_loss | 2.5 → 2.85，反向上升 |
| val/dfl_loss | 1.7 → 2.02，上升 |
| metrics/mAP50 | 30 轮后平稳在 0.20~0.24 |
| metrics/mAP50-95 | 50 轮后停在 0.06 |

### 2.4 混淆矩阵关键数据

- overripe_fruit 一整行为空（模型未成功预测出 overripe 类）
- 真实 overripe → 预测为 mature_fruit：0.29
- 真实 overripe → 漏检为 background：0.71
- background → 误判为 mature_fruit：0.87
- immature 漏检率：0.81

---

## 3. v5 训练（扩充数据 + 加大模型 + 提升分辨率）

### 3.1 数据集

| 项目 | v4 | v5 |
|---|---|---|
| 训练集图片 | 258 | 510 |
| 验证集图片 | 19 | 43 |
| 测试集图片 | 12 | 23 |
| mature_fruit 实例 | 3504 | 3943 |
| immature_fruit 实例 | 1275 | 1559 |
| overripe_fruit 实例 | 168 | 382 |

数据集路径：`data/labeled_v5/`

### 3.2 训练配置

| 参数 | v4_aug | v5 |
|---|---|---|
| model | yolo11n | yolo11s |
| imgsz | 640 | 960 |
| epochs | 150 | 200 |
| patience | 0 | 30 |
| cos_lr | False | True |
| close_mosaic | 15 | 20 |
| batch | 16 | 16 |
| device | 0 | 0 |

脚本：`scripts/train_v5.py`  
输出目录：`models/train_v5/`

### 3.3 训练结果

| 指标 | 数值 |
|---|---|
| 训练状态 | EarlyStopping 在第 63 轮触发，best 在第 33 轮 |
| 模型规模 | YOLO11s，9,413,961 参数，21.3 GFLOPs |
| mAP50 (all) | 0.365 |
| mAP50-95 (all) | 0.131 |
| Precision | 0.496 |
| Recall | 0.387 |
| immature_fruit mAP50 | 0.315 |
| mature_fruit mAP50 | 0.394 |
| overripe_fruit mAP50 | 0.385 |

验证集 per-class 详细：

| 类别 | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 43 | 630 | 0.496 | 0.387 | 0.365 | 0.131 |
| immature_fruit | 28 | 284 | 0.552 | 0.271 | 0.315 | 0.108 |
| mature_fruit | 29 | 304 | 0.460 | 0.463 | 0.394 | 0.122 |
| overripe_fruit | 20 | 42 | 0.476 | 0.429 | 0.385 | 0.162 |

推理速度：preprocess 0.1ms / inference 3.5ms / postprocess 2.2ms per image

### 3.4 训练曲线观察

| 曲线 | 走势 |
|---|---|
| train/*_loss | 全部平滑下降 |
| val/box_loss | 3.3 → 2.4（33 轮）→ 微升至 2.5 |
| val/cls_loss | 6.0 → 3.0 平稳 |
| val/dfl_loss | 4.0 → 2.3 → 微升 |
| metrics/mAP50 | 上升至 0.36 后平稳 |
| metrics/mAP50-95 | 上升至 0.13 后平稳 |

### 3.5 混淆矩阵关键数据

| 真实 \ 预测 | immature | mature | overripe | background（漏检） |
|---|---|---|---|---|
| immature | 0.26 | 0.13 | 0.01 | 0.60 |
| mature | 0.04 | 0.50 | 0.02 | 0.43 |
| overripe | 0.10 | 0.33 | 0.33 | 0.24 |

- background → 误判为 mature_fruit：0.63
- background → 误判为 immature_fruit：0.30
- background → 误判为 overripe_fruit：0.08

---

## 4. 三版本对比

| 指标 | v4 | v4_aug | v5 |
|---|---|---|---|
| 训练集图片 | 258 | 258 | 510 |
| 验证集图片 | 19 | 19 | 43 |
| 测试集图片 | 12 | 12 | 23 |
| 模型 | yolo11n | yolo11n | yolo11s |
| imgsz | 640 | 640 | 960 |
| best epoch | 27 | 27 | 33 |
| 停止方式 | EarlyStop @ 57 | 跑满 150 | EarlyStop @ 63 |
| mAP50 | 0.239 | 0.244 | 0.365 |
| mAP50-95 | 0.0785 | 0.084 | 0.131 |
| Precision | 0.401 | 0.401 | 0.496 |
| Recall | 0.256 | 0.256 | 0.387 |
| immature mAP50 | 0.254 | — | 0.315 |
| mature mAP50 | 0.381 | — | 0.394 |
| overripe mAP50 | 0.083 | — | 0.385 |

---

## 5. 输出文件清单

每个训练目录下包含：

```
models/train_vX/
├── weights/
│   ├── best.pt          # 验证集最优权重
│   └── last.pt          # 最后一轮权重
├── results.csv          # 每轮指标表格
├── results.png          # 训练曲线（10 张子图）
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── labels.jpg           # 标签分布
├── train_batch*.jpg     # 训练 batch 可视化
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── args.yaml            # 完整训练参数
```
