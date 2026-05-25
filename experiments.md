# 训练实验记录

本文件记录 v4 / v4_aug / v5 / v6 / v6_aug 几次训练的数据集、配置与结果。

类别：`immature_fruit` / `mature_fruit` / `overripe_fruit`（nc=3）

口径约定：所有 Precision / Recall / 单类 mAP50 都按 `best.pt` 对应轮取值。`best.pt` 由 Ultralytics 默认 `fitness = 0.1·mAP50 + 0.9·mAP50-95` 选出，等同于 mAP50-95 最优轮。

---

## 1. v4（基线）

### 1.1 数据集

| 项目 | 数值 |
|---|---|
| 训练集图片 | 258 |
| 验证集图片 | 19 |
| 测试集图片 | 12 |
| mature_fruit 实例 | 3504 |
| immature_fruit 实例 | 1275 |
| overripe_fruit 实例 | 168 |

路径：`data/labeled_v4/`

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
| 训练状态 | EarlyStop @ 57，best @ 27 |
| mAP50 (all) | 0.239 |
| mAP50-95 (all) | 0.0785 |
| Precision | 0.401 |
| Recall | 0.256 |
| immature_fruit mAP50 | 0.254 |
| mature_fruit mAP50 | 0.381 |
| overripe_fruit mAP50 | 0.083 |

输出：`models/train_v4/`

### 1.4 现象

overripe 类几乎学不到，mAP50 只有 0.083。三类样本量比约 21:8:1，模型偏向 mature。

---

## 2. v4_aug（关闭早停 + 强数据增强）

数据集与 v4 一致。

### 2.1 配置变化

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
输出：`models/train_v4_aug/`

### 2.2 训练结果

| 指标 | 数值 |
|---|---|
| 训练状态 | 跑满 150 轮，best @ 27 附近 |
| mAP50 (all) | 0.244 |
| mAP50-95 (all) | 0.0894 |
| Precision | 0.401 |
| Recall | 0.256 |

### 2.3 训练曲线

| 曲线 | 走势 |
|---|---|
| train/box_loss | 2.8 → 1.3 |
| train/cls_loss | 4.0 → 0.8 |
| val/box_loss | 2.5 → 2.85（反向上升） |
| val/dfl_loss | 1.7 → 2.02（上升） |
| metrics/mAP50 | 30 轮后停在 0.20~0.24 |
| metrics/mAP50-95 | 50 轮后停在 0.06 |

### 2.4 混淆矩阵

- overripe_fruit 整行为空（未成功预测出 overripe）
- 真实 overripe → 预测 mature：0.29
- 真实 overripe → 漏检 background：0.71
- background → 误判 mature：0.87
- immature 漏检率：0.81

### 2.5 现象

强增强没能救 overripe，问题在数据本身，不在增强强度。

---

## 3. v5（扩数据 + 加大模型 + 提分辨率）

### 3.1 数据集

| 项目 | v4 | v5 |
|---|---:|---:|
| 训练集图片 | 258 | 510 |
| 验证集图片 | 19 | 43 |
| 测试集图片 | 12 | 23 |
| mature_fruit 实例 | 3504 | 3943 |
| immature_fruit 实例 | 1275 | 1559 |
| overripe_fruit 实例 | 168 | 382 |

路径：`data/labeled_v5/`

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

脚本：`scripts/train_v5.py`
输出：`models/train_v5/`

### 3.3 训练结果

| 指标 | 数值 |
|---|---|
| 训练状态 | EarlyStop @ 63，best @ 33 |
| 模型规模 | YOLO11s，9,413,961 参数，21.3 GFLOPs |
| mAP50 (all) | 0.364 |
| mAP50-95 (all) | 0.131 |
| Precision | 0.495 |
| Recall | 0.387 |
| immature_fruit mAP50 | 0.315 |
| mature_fruit mAP50 | 0.394 |
| overripe_fruit mAP50 | 0.385 |

验证集 per-class：

| 类别 | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 43 | 630 | 0.495 | 0.387 | 0.364 | 0.131 |
| immature_fruit | 28 | 284 | 0.552 | 0.271 | 0.315 | 0.108 |
| mature_fruit | 29 | 304 | 0.460 | 0.463 | 0.394 | 0.122 |
| overripe_fruit | 20 | 42 | 0.476 | 0.429 | 0.385 | 0.162 |

推理速度：preprocess 0.1ms / inference 3.5ms / postprocess 2.2ms。

### 3.4 训练曲线

| 曲线 | 走势 |
|---|---|
| train/*_loss | 全部平滑下降 |
| val/box_loss | 3.3 → 2.4（33 轮）→ 微升至 2.5 |
| val/cls_loss | 6.0 → 3.0 平稳 |
| val/dfl_loss | 4.0 → 2.3 → 微升 |
| metrics/mAP50 | 上升至 0.36 后平稳 |
| metrics/mAP50-95 | 上升至 0.13 后平稳 |

### 3.5 混淆矩阵

归一化：

| 真实 \ 预测 | immature | mature | overripe | background |
|---|---:|---:|---:|---:|
| immature | 0.26 | 0.13 | 0.01 | 0.60 |
| mature | 0.04 | 0.50 | 0.02 | 0.43 |
| overripe | 0.10 | 0.33 | 0.33 | 0.24 |

原始计数（对角线为正确）：

| 类别 | 正确 | 漏检 background | 误检 mature | 误检 immature |
|---|---:|---:|---:|---:|
| immature_fruit | 73 | 171 | 36 | — |
| mature_fruit | 153 | 132 | — | 12 |
| overripe_fruit | 14 | 10 | — | — |

background → mature：0.63；background → immature：0.30；background → overripe：0.08。

### 3.6 v5 输出图解读

**F1 / P / R / PR 四条曲线**

- F1：所有类在 conf=0.251 附近峰值 0.42，与 YOLO 默认 0.25 接近。峰值只到 0.42 反映模型整体偏弱。
- P：conf 高时 Precision→1，但 conf=0.787 时 Recall 已经很低，没有实际意义。
- R：conf→0 时 Recall 峰值约 0.70，conf>0.6 后基本归零，说明很多预测置信度本来就不高。
- PR：曲线整体偏低，右上角不饱满，三类 AP@0.5 immature 0.315 / mature 0.394 / overripe 0.385。

**混淆矩阵**

immature 漏检 171（远超正确 73），是当前最大单点问题。背景误检里 mature 占大头（106 次），模型倾向把叶簇、光斑、阴影里的圆形结构当熟果。immature → mature 误判 0.13，反向只有 0.04，颜色过渡区间整体偏向"判熟"。

**labels.jpg**

实例分布 3943 / 1559 / 382，mature 是 overripe 的 10 倍多。bbox 宽高大多集中在 normalized < 0.1 的区间，目标普遍很小，imgsz=960 也只能部分缓解。新补的图片可能引入更多模糊小 immature，把整体任务变难，所以 v4→v5 大幅补图后 immature 的 Recall 没有同比上涨。

**results.png**

训练 loss 持续下降，验证 box/dfl loss 在 30 轮触底后微升，Precision 后期波动，mAP50 在 25–40 轮达到峰值后平稳。前 33 轮学到主要东西，之后开始走泛化下坡，EarlyStop @ 63 合理。

**val_batch**

预测置信度普遍 0.3–0.6，模型不算自信。密集树叶区域出现叶簇被识别为果子、同一果子被重叠预测；颜色过渡果存在 mature ↔ immature 互判，少量 overripe 被吞进 mature 或 immature。

### 3.7 现象

overripe 类从 v4 的 0.083 恢复到 0.385，整体 mAP50 从 0.244 翻到 0.365。剩余四个短板：immature 漏检最严重；背景误检 mature 最严重；颜色过渡区间倾向判熟；小目标多。

---

## 4. v6（标注口径统一 + 框贴合 + 加正则化）

### 4.1 数据集

v6 是一次标注口径重构：

1. 统一 mature_fruit 的判定标准，减少颜色过渡区间前后不一致。
2. 把过大的 bbox 改小，框紧果实本体，减少叶片/树枝/天空被框入。
3. Roboflow 重新导出，`data/labeled_v6/data.yaml`。

| 项目 | v5 | v6 |
|---|---:|---:|
| 训练集图片 | 425 | 510 |
| 验证集图片 | 43 | 43 |
| 测试集图片 | 23 | 23 |
| 训练集总实例 | 5884 | 6363 |
| immature_fruit 训练实例 | 1559 | 1638 |
| mature_fruit 训练实例 | 3943 | 4485 |
| overripe_fruit 训练实例 | 382 | 240 |
| 验证集总实例 | 630 | 575 |
| 测试集总实例 | 623 | 487 |

训练集 mature : immature : overripe 约 18.7 : 6.8 : 1。overripe 从 382 降到 240，是因为部分不够典型的 overripe 在统一标准后被重新归类或移除。

bbox 中位面积：训练集 0.0048 → 0.0039，验证 0.0041 → 0.0037，测试 0.0018 → 0.0017。框更贴合目标后背景噪声减少，但任务变成"小目标精定位"，对 mAP50-95 更敏感。

### 4.2 训练配置

| 参数 | v5 | v6 |
|---|---|---|
| model | yolo11s.pt | yolo11s.pt |
| data | labeled_v5/data.yaml | labeled_v6/data.yaml |
| imgsz | 960 | 960 |
| epochs | 200 | 200 |
| patience | 30 | 40 |
| batch | 16 | 16 |
| cos_lr | True | True |
| cls | 0.7 | 0.7 |
| weight_decay | 0.0005 | 0.001 |
| dropout | 0.0 | 0.1 |
| close_mosaic | 20 | 20 |

脚本：`scripts/train_v6.py`
输出：`models/train_v6/`

### 4.3 训练结果

| 指标 | v5 (best.pt) | v6 (best.pt) |
|---|---:|---:|
| 停止轮数 | 63 | 84 |
| best epoch（mAP50-95 最优） | 33 | 44 |
| mAP50 | 0.364 | 0.341 |
| mAP50-95 | 0.131 | 0.133 |
| Precision | 0.495 | 0.414 |
| Recall | 0.387 | 0.394 |
| 最后一轮 mAP50 | 0.324 | 0.313 |
| 最后一轮 mAP50-95 | 0.112 | 0.114 |

best 权重 per-class：

| 类别 | v5 AP@0.5 | v6 AP@0.5 |
|---|---:|---:|
| immature_fruit | 0.315 | 0.289 |
| mature_fruit | 0.394 | 0.500 |
| overripe_fruit | 0.385 | 0.232 |
| all | 0.364 | 0.341 |

总 mAP50 没超过 v5，但 mAP50-95 略升。mature AP 从 0.394 涨到 0.500 是最明确的收益。immature 小幅下降，overripe 明显下降——overripe 训练实例从 382 砍到 240，验证集只有 23 个，分母小，波动放大。

### 4.4 训练曲线

| 曲线 | 走势 |
|---|---|
| train/box_loss | 2.2 → 1.28 |
| train/cls_loss | 4.0 → 1.13 |
| train/dfl_loss | 2.0 → 1.25 |
| val/box_loss | 前 10 轮降到 2.25，后期升到 2.5 |
| val/cls_loss | 6.0 → 3.0，后期在 3.1~3.3 波动 |
| val/dfl_loss | 前期降到 2.0，后期升到 2.2~2.3 |
| metrics/mAP50 | 10 轮后进入 0.28~0.35 平台 |
| metrics/mAP50-95 | 20 轮后在 0.10~0.13 波动 |

训练 loss 继续下降，验证 box/dfl loss 在前期触底后回升，过拟合仍在。`dropout=0.1` 和 `weight_decay=0.001` 没解决泛化问题，只是把训练拖长了。

### 4.5 混淆矩阵

原始计数：

| 真实 \ 预测 | immature | mature | overripe | background |
|---|---:|---:|---:|---:|
| immature | 71 | 24 | 0 | 118 |
| mature | 28 | 193 | 4 | 114 |
| overripe | 3 | 10 | 3 | 7 |
| background | 53 | 157 | 2 | — |

归一化：

| 真实 \ 预测 | immature | mature | overripe | background |
|---|---:|---:|---:|---:|
| immature | 0.33 | 0.11 | 0.00 | 0.55 |
| mature | 0.08 | 0.57 | 0.01 | 0.34 |
| overripe | 0.13 | 0.43 | 0.13 | 0.30 |

mature 正确比例 0.50 → 0.57，漏检 0.43 → 0.34，是 mature 标准统一最直接的收益。immature 正确比例 0.26 → 0.33，但漏检仍 0.55。overripe 正确比例 0.33 → 0.13，且 0.43 被判成 mature，过熟/成熟边界仍最弱。背景误检里 mature 占 0.74。

### 4.6 现象

mature 的收益是真实的（AP 涨了 0.106，正确比例涨了 0.07，漏检比例降了 0.09）。overripe 的下降同时受两个因素影响：训练实例从 382 砍到 240，以及 dropout=0.1 + weight_decay=0.001 让模型更保守。两个因素叠在一起没法拆开，所以 v6_aug 把训练参数回退做了一次对照。

预测可视化（`runs/detect/predict3/`）：mature 检出最稳，置信度多在 0.3~0.7；框比 v5 更贴本体；immature 只有颜色非常绿、形态清晰时稳定；overripe 几乎不被预测，多数被吸到 mature；密集树冠和强光区域仍有 mature 误检。

**关于 mAP50 下降的判读**：mAP50 由 IoU≥0.5 判命中，IoU 同时受框大小和位置影响。v5 的框普遍偏大，预测只要大致压到那一片区域 IoU 就容易过 0.5；v6 把框收紧后，真实框变小，同一份预测可能因为偏移几个像素掉到 0.5 以下。所以 v6 mAP50 的下降里有一部分是评测标准变严了，不全是模型退步。mAP50-95 对定位更敏感，v6 反而略升，这一块才是真实定位能力的变化。

---

## 5. v6_aug（v6 数据 + 回退正则化）

### 5.1 目的

v6 同时改了两件事：数据更严格 + 正则化更强。指标下降时不能直接判断是哪一边导致的。v6_aug 用同一份 `data/labeled_v6/`，把训练参数回退到接近 v5 的水平，单独看正则化的影响。

### 5.2 训练配置

| 参数 | v6 | v6_aug |
|---|---:|---:|
| data | labeled_v6/data.yaml | labeled_v6/data.yaml |
| model | yolo11s.pt | yolo11s.pt |
| imgsz | 960 | 960 |
| patience | 40 | 40 |
| cls | 0.7 | 0.7 |
| weight_decay | 0.001 | 0.0005 |
| dropout | 0.1 | 0.0 |

脚本：`scripts/train_v7.py`（输出改名为 `train_v6_aug`）
输出：`models/train_v6_aug/`

### 5.3 训练结果

| 指标 | v6 (best.pt) | v6_aug (best.pt) |
|---|---:|---:|
| 停止轮数 | 84 | 84 |
| best epoch（mAP50-95 最优） | 44 | 44 |
| mAP50 | 0.341 | 0.356 |
| mAP50-95 | 0.133 | 0.138 |
| Precision | 0.414 | 0.453 |
| Recall | 0.394 | 0.383 |
| 最后一轮 mAP50 | 0.313 | 0.323 |
| 最后一轮 mAP50-95 | 0.114 | 0.117 |

best 权重 per-class：

| 类别 | v6 AP@0.5 | v6_aug AP@0.5 | Δ |
|---|---:|---:|---:|
| immature_fruit | 0.289 | 0.309 | +0.020 |
| mature_fruit | 0.500 | 0.557 | +0.057 |
| overripe_fruit | 0.232 | 0.204 | -0.028 |
| all | 0.341 | 0.356 | +0.015 |

v6_aug 在 mAP50 / mAP50-95 / Precision 三项都比 v6 好。v6 的 `dropout=0.1` 加 `weight_decay=0.001` 对当前小目标小数据集偏强，让模型过度保守；去掉 dropout 后学得更充分。但 overripe 的问题不是去掉 dropout 能解决的，瓶颈在样本数和类别边界。

### 5.4 训练曲线

| 曲线 | 走势 |
|---|---|
| train/box_loss | 2.2 → 1.29 |
| train/cls_loss | 4.0 → 1.13 |
| train/dfl_loss | 2.0 → 1.24 |
| val/box_loss | 前 10 轮降到 2.25，后期升到 2.5 |
| val/cls_loss | 前期降到 2.8~3.0，后期回到 3.2 |
| val/dfl_loss | 前期降到 2.0，后期升到 2.3 |
| metrics/mAP50 | 第 44 轮峰值 0.356，后期在 0.29~0.33 波动 |
| metrics/mAP50-95 | 第 44 轮峰值 0.138，后期在 0.10~0.12 波动 |

mAP50 和 mAP50-95 双峰值都集中在第 44 轮，比 v6 的 mAP50 峰值（第 78 轮）更早。去掉强正则化后模型更快收敛到主要模式。

### 5.5 混淆矩阵

原始计数：

| 真实 \ 预测 | immature | mature | overripe | background |
|---|---:|---:|---:|---:|
| immature | 80 | 25 | 1 | 107 |
| mature | 37 | 202 | 4 | 96 |
| overripe | 0 | 12 | 4 | 7 |
| background | 89 | 145 | 7 | — |

归一化：

| 真实 \ 预测 | immature | mature | overripe | background |
|---|---:|---:|---:|---:|
| immature | 0.38 | 0.12 | 0.00 | 0.50 |
| mature | 0.11 | 0.60 | 0.01 | 0.28 |
| overripe | 0.00 | 0.52 | 0.17 | 0.30 |

immature 正确 0.33 → 0.38，mature 正确 0.57 → 0.60，mature 漏检 0.34 → 0.28，背景误检 mature 从 157 降到 145。overripe 正确比例 0.13 → 0.17，但 AP 从 0.232 降到 0.204，且 0.52 被预测成 mature——模型不是找不到 overripe，是排序和置信度不稳，并且大部分被 mature 吞掉。

### 5.6 现象

- v6 的数据清洗方向是对的：mature 类 AP 从 0.394 一路涨到 0.557，正确比例和漏检比例都改善。
- v6 的强正则化偏强：v6_aug 回退后总 mAP 和 Precision 都回升。当前阶段保留 `dropout=0.0` + `weight_decay=0.0005`。
- overripe 是结构性短板，不靠参数解决：训练集只剩 240 实例，验证集只有 23 实例，本身波动大；且和 mature 的视觉边界仍最弱。
- mature 已经是当前最稳定的类别，再补 mature 收益递减。

### 5.7 下一步

1. **数据**：把 overripe 训练实例补到 500~600，重点是黑斑、干瘪、破损、腐烂这类典型样本，不要用普通深红熟果硬凑。
2. **标注复查**：mature / overripe 边界（斑点、局部破损、深红但未腐烂）走一遍统一标准，避免 overripe 被当成 mature 的子集；模糊不可辨认的小果继续不画框。
3. **保留 v6 的框贴合标准**，不再把大片背景框进目标。
4. **训练参数固定**：`yolo11s` + `imgsz=960` + `dropout=0.0` + `weight_decay=0.0005` + `patience=40`。overripe 补足前不上 yolo11m，否则过拟合更严重。
5. **推理阈值**：F1 峰值在 conf≈0.25 附近，少漏检走 0.18，少误检走 0.30~0.35。

---

## 6. 五版本对比

口径：所有 Precision / Recall / 单类 mAP50 取 best.pt 对应轮（mAP50-95 最优轮）。

| 指标 | v4 | v4_aug | v5 | v6 | v6_aug |
|---|---:|---:|---:|---:|---:|
| 训练集图片 | 258 | 258 | 425 | 510 | 510 |
| 验证集图片 | 19 | 19 | 43 | 43 | 43 |
| 测试集图片 | 12 | 12 | 23 | 23 | 23 |
| 训练集实例 | 4947 | 4947 | 5884 | 6363 | 6363 |
| 模型 | yolo11n | yolo11n | yolo11s | yolo11s | yolo11s |
| imgsz | 640 | 640 | 960 | 960 | 960 |
| best epoch | 27 | 21 | 33 | 44 | 44 |
| 停止 | EarlyStop @ 57 | 跑满 150 | EarlyStop @ 63 | EarlyStop @ 84 | EarlyStop @ 84 |
| mAP50 | 0.239 | 0.244 | 0.364 | 0.341 | 0.356 |
| mAP50-95 | 0.0785 | 0.0894 | 0.131 | 0.133 | 0.138 |
| Precision | 0.401 | 0.401 | 0.495 | 0.414 | 0.453 |
| Recall | 0.256 | 0.256 | 0.387 | 0.394 | 0.383 |
| immature mAP50 | 0.254 | — | 0.315 | 0.289 | 0.309 |
| mature mAP50 | 0.381 | — | 0.394 | 0.500 | 0.557 |
| overripe mAP50 | 0.083 | — | 0.385 | 0.232 | 0.204 |

---

## 7. 输出文件清单

```
models/train_vX/
├── weights/
│   ├── best.pt          # 验证集最优权重（按 fitness）
│   └── last.pt          # 最后一轮权重
├── results.csv          # 每轮指标表格
├── results.png          # 训练曲线
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── labels.jpg           # 标签分布
├── train_batch*.jpg
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── args.yaml            # 完整训练参数
```
