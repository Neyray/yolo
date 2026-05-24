# 训练实验记录

本文件记录 v4 / v4_aug / v5 / v6 几次训练的数据集、配置与结果。

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

### 3.6 v5 输出图详细分析

补一份 `models/train_v5/` 下所有评估图的解读，单图都不能独立下结论，要放在一起看。

#### 3.6.1 四条曲线：F1 / P / R / PR

`BoxF1_curve.png` 显示，所有类别在 conf=0.251 附近 F1 达到峰值 0.42。这意味着推理阶段 conf 设到 0.25 左右最稳，和 YOLO 默认值接近。F1 峰值只有 0.42 也直接反映模型整体还偏弱——三类里 mature_fruit 相对最稳，immature 和 overripe 都偏低。

`BoxP_curve.png` 的趋势正常：低 conf 时框多但准度差，高 conf 时只剩极少高把握的框，所以右上角 Precision 接近 1.0。但 conf=0.787 时 Precision=1.0 这个点没有实际意义，因为此时 Recall 已经很低，大量果子被漏掉。对当前任务来说，把 conf 推到 0.5 以上没有好处。

`BoxR_curve.png` 显示 conf→0 时 Recall 峰值约 0.70，conf 增大后快速下降，0.6 以后基本归零。也就是说很多预测框的置信度本来就不高，推理时把阈值设太严会丢掉一大批真实目标。如果偏向少漏检，可以试 conf=0.15；偏向少误检，可以试 conf=0.35。

`BoxPR_curve.png` 是最核心的一张，曲线整体偏低，右上角不饱满，说明模型综合能力一般。三类 AP@0.5：immature 0.315 / mature 0.394 / overripe 0.385，total 0.365。值得注意的是 overripe 在数据量最少的情况下 AP 反而比 immature 略高，这间接说明 immature 的漏检比 overripe 更严重。

#### 3.6.2 混淆矩阵：漏检远比误检严重

`confusion_matrix.png` 的原始计数（对角线代表预测正确数）：

| 类别 | 正确 | 漏检（被判为 background） | 误检 mature | 误检 immature |
|---|---:|---:|---:|---:|
| immature_fruit | 73 | 171 | 36 | — |
| mature_fruit | 153 | 132 | — | 12 |
| overripe_fruit | 14 | 10 | — | — |

immature 的漏检数（171）已经远超正确预测数（73），是当前最大的单点问题。背景被误判为果子的情况里 mature 占大头（106 次），说明模型把树叶、光斑、阴影里圆形结构当成熟果的频率最高。

`confusion_matrix_normalized.png` 是同样的信息按比例呈现：immature 正确率 0.26，mature 0.50，overripe 0.33。对应到背景漏检列，immature 真实目标被直接漏掉的比例高达 0.60，这是 Recall 偏低的主要来源。immature ↔ mature 之间还有非对称错判——immature 被判为 mature 0.13，反向只有 0.04，说明模型在颜色过渡区间更倾向于"判熟"。

#### 3.6.3 labels.jpg：标注分布暴露两个结构性问题

实例数分布是 3943 / 1559 / 382（mature / immature / overripe），mature 是 overripe 的 10 倍多，模型自然会偏向 mature。这是为什么 overripe 在 v5 已经从 0.083 涨到 0.385，但还有继续上涨空间——只要 overripe 实例继续补到接近 immature 的量级，它的 AP 还会再走高。

右下角的 bbox 宽高分布同样关键：训练集大多数框都集中在 normalized width/height 小于 0.1 的区间，也就是说目标普遍很小。即便 v5 已经把 imgsz 提到 960，小目标对 YOLO 仍然是个慢性病。这也解释了一个反直觉现象——v4→v5 大量补图后 immature 的 Recall 反而没怎么涨，是因为新补的图里可能引入了更多小且模糊的 immature 目标，整体抬高了任务难度。

#### 3.6.4 results.png：训练学到了东西，但泛化在退步

训练集三条 loss（box / cls / dfl）从开始到结束都在平滑下降，说明模型确实在学。但验证集 box_loss 在第 30 轮左右触底 2.4 后微升到 2.5，dfl_loss 同样有轻微反弹。Precision 后期波动，Recall 没有进一步抬升，mAP50 在 25–40 轮之间达到峰值后基本平稳。

整体走势不是"没学会"，而是"前 33 轮学到了主要的东西，之后开始走泛化下坡"。第 33 轮 best、第 63 轮早停的触发是合理的——继续训只会让训练集 loss 接着降而验证集变差。

#### 3.6.5 val_batch 可视化：直观问题确认

`val_batch0/1_labels.jpg` 是人工标注的 ground truth，能看出几个隐性问题：树冠和远处区域框非常密集，部分图里小框堆叠严重，类别在颜色过渡区域（半红半绿）确实存在主观性。

`val_batch0/1_pred.jpg` 是 best.pt 在同一组图上的预测，置信度普遍集中在 0.3–0.6，模型不算很自信。密集树叶区域可以观察到典型的小目标误检——把叶簇当成果子，且会有同一果子被多次重叠预测。部分原本是 mature 的果子被预测成 immature，少量 overripe 被预测成 mature 或 immature，这和混淆矩阵反映的非对角错判是一致的。

#### 3.6.6 综合结论与下一轮方向

v5 这一轮的核心收益是 overripe 类从"模型放弃"恢复到了"和其他类同水平"，整体 mAP50 翻了 1.5 倍。剩余问题集中在四点：immature 漏检最严重（171 个，超过正确数 73）；mature 误检最严重（背景被判 mature 106 次）；类间存在不对称混淆，颜色过渡区域偏向判熟；小目标整体偏多，YOLO 不友好。

下一轮（v6）的改动重点应该放在数据侧：继续把 overripe 补到 600+ 实例，使三类比例从当前的 10 : 4 : 1 缩到更平衡的量级；统一颜色过渡区间的标注口径，模糊或不可分辨的小目标不强行画框，避免拉高任务难度；视觉上明显腐烂、黑斑、干瘪、破损的优先标 overripe，而不是用普通红果硬凑数。训练侧次优先尝试 dropout=0.1、weight_decay=0.001、label_smoothing=0.1 抑制过拟合和类不平衡的影响；模型规模目前是 yolo11s，等数据进一步扩到 700+ 张再考虑 yolo11m，否则会更严重过拟合。

---

## 4. v6 训练（标注口径统一 + 框更贴合目标 + 正则化）

### 4.1 数据集

v6 不是单纯换训练参数，而是一次数据标注口径重构：

1. 统一 `mature_fruit` 的判断标准，减少颜色过渡区间的前后不一致。
2. 修正一部分过大的 bbox，让框更贴近果实本体，减少叶片、树枝、天空等背景被框入目标。
3. 使用 Roboflow v6 导出数据集，训练脚本指向 `data/labeled_v6/data.yaml`。

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

数据集路径：`data/labeled_v6/`

v6 的实例分布更偏向 mature：训练集 mature:immature:overripe 约为 18.7 : 6.8 : 1。overripe 从 382 降到 240，说明这次统一标准后，部分原本偏模糊或不够典型的 overripe 被重新归类或移除。这个变化让 mature 的学习目标更清晰，但也让 overripe 的样本不足重新成为主要短板。

bbox 尺寸也整体变小：训练集框面积中位数从 v5 的 0.0048 降到 v6 的 0.0039，valid 从 0.0041 降到 0.0037，test 从 0.0018 降到 0.0017。框更贴合目标后，模型看到的背景噪声减少，但任务也更接近“小目标精定位”，对 mAP50-95 的影响会比对 mAP50 更敏感。

### 4.2 训练配置

| 参数 | v5 | v6 |
|---|---|---|
| model | yolo11s.pt | yolo11s.pt |
| data | `data/labeled_v5/data.yaml` | `data/labeled_v6/data.yaml` |
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
输出目录：`models/train_v6/`

### 4.3 训练结果

`best.pt` 按 Ultralytics 默认 fitness 保存，核心更偏向 mAP50-95；因此下面同时列出 mAP50 最优轮和 mAP50-95 最优轮。

| 指标 | v5 | v6 |
|---|---:|---:|
| 停止轮数 | 63 | 84 |
| mAP50 最优 epoch | 42 | 78 |
| mAP50 最优值 | 0.36573 | 0.34884 |
| mAP50 最优时 Precision | 0.41178 | 0.50434 |
| mAP50 最优时 Recall | 0.39633 | 0.33558 |
| mAP50-95 最优 epoch | 33 | 44 |
| mAP50-95 最优值 | 0.13100 | 0.13348 |
| 最后一轮 mAP50 | 0.32389 | 0.31274 |
| 最后一轮 mAP50-95 | 0.11233 | 0.11410 |

从总指标看，v6 的 mAP50 比 v5 低一点，但 mAP50-95 略高。这个组合符合“框更贴合目标”的预期：宽松 IoU=0.5 下整体检测能力没有明显超过 v5，但严格定位指标略有改善。Precision 从 0.41 提到 0.50，说明误检减少；Recall 从 0.40 降到 0.34，说明漏检增加，尤其是更小、更严格的目标更容易被漏掉。

best 权重对应的 `BoxPR_curve.png`：

| 类别 | v5 AP@0.5 | v6 AP@0.5 |
|---|---:|---:|
| immature_fruit | 0.315 | 0.289 |
| mature_fruit | 0.394 | 0.500 |
| overripe_fruit | 0.385 | 0.232 |
| all | 0.365 | 0.341 |

成熟果的 AP 从 0.394 提到 0.500，是 v6 最明确的收益，说明统一 mature 标准和收紧框大小确实帮模型学到了更稳定的 mature 特征。immature 小幅下降，overripe 明显下降，主要原因不是训练参数，而是 v6 数据里 overripe 样本更少，且标准更严格后可学习样本变得更稀疏。

### 4.4 训练曲线观察

| 曲线 | 走势 |
|---|---|
| train/box_loss | 2.2 → 1.28，持续下降 |
| train/cls_loss | 4.0 → 1.13，持续下降 |
| train/dfl_loss | 2.0 → 1.25，持续下降 |
| val/box_loss | 前 10 轮下降到约 2.25，之后缓慢升到 2.5 |
| val/cls_loss | 从 6.0 快速降到 3.0 左右，后期在 3.1~3.3 波动 |
| val/dfl_loss | 前期降到 2.0 左右，后期升到 2.2~2.3 |
| metrics/mAP50 | 10 轮后进入 0.28~0.35 平台，最高在第 78 轮 |
| metrics/mAP50-95 | 20 轮后基本在 0.10~0.13 波动，最高在第 44 轮 |

v6 的训练集 loss 继续平滑下降，但验证集 box/dfl loss 在早期触底后上升，说明过拟合仍然存在。`dropout=0.1` 和 `weight_decay=0.001` 没有完全解决泛化问题，只是让模型后期还能多跑一些轮次；真正决定效果的仍然是标注质量和类别分布。

### 4.5 混淆矩阵关键数据

原始计数：

| 真实 \ 预测 | immature | mature | overripe | background（漏检） |
|---|---:|---:|---:|---:|
| immature | 71 | 24 | 0 | 118 |
| mature | 28 | 193 | 4 | 114 |
| overripe | 3 | 10 | 3 | 7 |
| background（误检） | 53 | 157 | 2 | — |

归一化矩阵：

| 真实 \ 预测 | immature | mature | overripe | background（漏检） |
|---|---:|---:|---:|---:|
| immature | 0.33 | 0.11 | 0.00 | 0.55 |
| mature | 0.08 | 0.57 | 0.01 | 0.34 |
| overripe | 0.13 | 0.43 | 0.13 | 0.30 |

v6 相比 v5，mature 的正确比例从 0.50 提到 0.57，漏检比例从 0.43 降到 0.34，这是 mature 标准统一后最有价值的改善。immature 正确比例从 0.26 提到 0.33，但漏检仍高达 0.55。overripe 正确比例从 0.33 降到 0.13，且 0.43 被判成 mature，说明当前 overripe 和 mature 的边界仍然是最弱环节。

背景误检仍主要集中在 mature：background → mature 为 157 次，占背景误检的 0.74。也就是说，虽然 mature 类本身更稳了，但模型仍容易把背景里的圆形亮斑、叶簇、红黄区域当成熟果。

### 4.6 综合结论与下一轮方向

v6 的核心收益不是总 mAP 上升，而是把数据问题拆清楚了：统一 mature 标准和收紧 bbox 后，mature_fruit 的 AP 明显上涨，框定位的严格指标也略有改善；但 overripe 样本减少后，过熟类 AP 和召回明显回落。v6 证明“标注口径一致性”对 mature 有直接帮助，同时也暴露了 overripe 不能只靠更严格标准，必须补充更多典型样本。

下一轮建议优先做数据侧，而不是继续堆训练参数：

1. 补充 overripe 到至少 500~600 个训练实例，尤其是黑斑、干瘪、破损、腐烂这些典型高风险样本。
2. 保持 v6 的 bbox 贴合标准，不要再把大面积背景框进目标。
3. 继续复查 mature/overripe 边界，避免“普通红熟果”被标成 overripe，也避免明显破损果被归入 mature。
4. 推理阈值建议从 `conf=0.18~0.25` 试起；如果更重视少漏检，用 0.18 附近，如果更重视少误检，用 0.25~0.35。
5. 在 overripe 样本补足前，不建议升级到 yolo11m；当前瓶颈主要是类别样本和边界定义，不是模型容量。

---

## 5. 四版本对比

| 指标 | v4 | v4_aug | v5 | v6 |
|---|---:|---:|---:|---:|
| 训练集图片 | 258 | 258 | 425 | 510 |
| 验证集图片 | 19 | 19 | 43 | 43 |
| 测试集图片 | 12 | 12 | 23 | 23 |
| 训练集实例 | 4947 | 4947 | 5884 | 6363 |
| 模型 | yolo11n | yolo11n | yolo11s | yolo11s |
| imgsz | 640 | 640 | 960 | 960 |
| best epoch（按 mAP50-95） | 27 | 21 | 33 | 44 |
| 停止方式 | EarlyStop @ 57 | 跑满 150 | EarlyStop @ 63 | EarlyStop @ 84 |
| mAP50（best.pt） | 0.239 | 0.244 | 0.365 | 0.341 |
| mAP50-95（best.pt） | 0.0785 | 0.0894 | 0.131 | 0.133 |
| Precision（mAP50 最优轮） | 0.452 | 0.356 | 0.412 | 0.504 |
| Recall（mAP50 最优轮） | 0.232 | 0.304 | 0.396 | 0.336 |
| immature mAP50 | 0.254 | — | 0.315 | 0.289 |
| mature mAP50 | 0.381 | — | 0.394 | 0.500 |
| overripe mAP50 | 0.083 | — | 0.385 | 0.232 |

---

## 6. 输出文件清单

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
