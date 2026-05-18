# 果实检测：YOLO 果实状态识别与停车风险评估

## 研究方向

当前方向：

> 面向停车安全的仰视果实检测：先用 YOLO 检测果实状态/类型，再评估车是否适合停在树下。

后续不继续把项目只当成"数果子"。旧的单类别模型可以保留作为基线模型，但主模型应改成多类别检测。

## 推荐类别

Roboflow 中使用这 3 个类别：

```yaml
nc: 3
names:
  - immature_fruit
  - mature_fruit
  - overripe_fruit
```

类别含义：

```text
immature_fruit: 未成熟果实
mature_fruit: 成熟果实
overripe_fruit: 过熟、破损、干枯或明显高风险果实
```

标注规则：

1. 每个可见的独立果实都框一个 bbox，避免漏标。
2. 多个果实重叠时，按"主体可见"原则分别框出。
3. 体积较小、绿色、明显未成熟的果实标为 `immature_fruit`。
4. 大小和颜色都接近正常成熟状态的果实标为 `mature_fruit`。
5. 发暗、破损、干枯、腐烂或明显过熟的果实标为 `overripe_fruit`。
6. 不确定的样本先记录下来，之后集中复查，避免前后标注标准不一致。

## 数据目录

从 Roboflow 导出 YOLO/Ultralytics 格式后，把新的多类别数据集放到这里：

```text
data/fruit_risk/data.yaml
data/fruit_risk/train/images
data/fruit_risk/train/labels
data/fruit_risk/valid/images
data/fruit_risk/valid/labels
data/fruit_risk/test/images
data/fruit_risk/test/labels
```

旧的单类别数据集如果继续保留，只作为基线对比使用。

## 安装依赖

```bash
cd /home/jerico/projects/fruit_detect
pip install -r requirements.txt
```

## 训练

默认训练命令：

```bash
cd /home/jerico/projects/fruit_detect
python scripts/train_v5.py
```

常用参数：

```bash
python scripts/train.py --weights yolo11s.pt --epochs 200 --batch 16 --name fruit_risk_yolo11s
python scripts/train.py --data /path/to/data.yaml --device cpu
```

默认输出模型位置：

```text
models/train_v5/weights/best.pt
```

## 检测并判断停车风险

```bash
cd /home/jerico/projects/fruit_detect
python scripts/detect_by_yolo.py data/raw/01.png
```

如果模型保存在其他位置：

```bash
FRUIT_MODEL_PATH=/path/to/best.pt python scripts/detect_by_yolo.py data/raw/01.png
```

YOLO 检测脚本现在分两步：

1. 检测果实类别和位置。
2. 根据检测结果计算风险分数，并输出停车建议。

当前风险权重：

```text
immature_fruit: 1.0
mature_fruit: 2.0
overripe_fruit: 5.0
```

停车建议规则：

```text
高风险：overripe >= 2，或 risk_score >= 12
中风险：overripe >= 1，或 mature >= 3，或 risk_score >= 6
低风险：没有检测到果实，或主要是未成熟果实
```


## Qwen-VL 对比方法

Qwen-VL 适合作为辅助分析或对比基线，不作为唯一方法。

使用前先设置 API Key：

```bash
export DASHSCOPE_API_KEY='your_key_here'
python scripts/detect_by_qwen.py data/raw/01.png
```


## 评估指标说明

YOLO 训练完成后会在 `models/train_vX/` 目录下输出四张曲线图和一份混淆矩阵，对应的核心指标含义如下。所有指标都在验证集（valid set）上计算。

### Precision（精确率）

```text
Precision = TP / (TP + FP)
```

含义：模型预测出的框里，有多少是真目标。
- TP（True Positive）：预测正确的框，与某个真实标注框 IoU ≥ 0.5 且类别一致。
- FP（False Positive）：预测错误的框，比如把树叶、阴影、树枝判成果子，或者把果子类别认错。

Precision 偏低，意味着误检多——模型"敢说"但说错的频率高。一般通过提高推理阈值 `conf` 来抬 Precision，但会以牺牲 Recall 为代价。

### Recall（召回率）

```text
Recall = TP / (TP + FN)
```

含义：真实存在的目标里，模型找到了多少。
- FN（False Negative）：漏检，真实有果子但模型没框出来。

Recall 偏低，意味着漏检严重——很多果子根本没被发现。常见原因有目标太小、被遮挡、类别样本不均衡、标注质量差等。对"树上果子检测"这种任务，Recall 通常比 Precision 更重要。

### F1

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

含义：Precision 和 Recall 的调和平均，两者只要有一边低，F1 就低。`BoxF1_curve.png` 会画出 F1 随 conf 阈值的变化，曲线峰值对应的 conf 通常是推理时的最佳阈值。

### mAP@0.5（IoU=0.5 下的平均精度）

mAP 是 mean Average Precision 的缩写，先对每个类别在不同 Recall 下计算精度，得到 PR 曲线下面积 AP，再对所有类别求平均。

`@0.5` 表示判定 TP 的 IoU 阈值是 0.5——只要预测框和真实框重叠面积占总面积的一半以上，就算位置基本正确。这是目标检测最常用的综合指标，对位置偏差容忍度较高。

### mAP@0.5:0.95（COCO 风格主指标）

把 IoU 阈值从 0.5 到 0.95 每隔 0.05 各算一次 mAP，再取平均，一共 10 个阈值。

比 mAP@0.5 严格得多，因为高 IoU 阈值要求框得很准。mAP@0.5:0.95 通常显著低于 mAP@0.5，能反映框的位置精度。论文里说的"mAP"如果没注明 IoU 一般指这个值。

### Per-class mAP

对每个类别单独算 mAP@0.5。如果某一类的 mAP 远低于其他类，要么是该类样本严重不足，要么是类别边界本身定义不清。比较 per-class mAP 是排查"哪类塌了"的最快方式。

### 混淆矩阵（Confusion Matrix）

行表示真实类别，列表示模型预测类别，多出来一行一列是 background（无目标）。

- 对角线：预测正确的数量（或比例）。
- 非对角线（类间）：类别混淆，比如真实是 mature 被预测成 immature。
- 最后一列 background：真实有目标但被漏检（看 Recall 偏低时主要看这里）。
- 最后一行 background：实际是背景被预测成果子（看 Precision 偏低时主要看这里）。

`confusion_matrix.png` 显示原始计数，`confusion_matrix_normalized.png` 按行归一化成比例，后者更适合看每一类的错误结构。

### 四张曲线图速读

| 文件 | 横轴 | 纵轴 | 看什么 |
|---|---|---|---|
| `BoxP_curve.png` | conf 阈值 | Precision | conf 升高时精度怎么涨 |
| `BoxR_curve.png` | conf 阈值 | Recall | conf 升高时召回怎么掉 |
| `BoxF1_curve.png` | conf 阈值 | F1 | 峰值对应最佳推理阈值 |
| `BoxPR_curve.png` | Recall | Precision | 曲线越靠右上越好，下方面积≈AP |

## 推荐实验设计

1. 基线实验：旧单类别 YOLO，只统计果实数量。
2. 主方法实验：多类别 YOLO，识别果实成熟度/状态。
3. 风险评估实验：YOLO 检测结果 + 规则化风险评分。
4. 可选对比实验：Qwen-VL 视觉分析。

停车风险指标可以自定义：

```text
risk_accuracy = 风险判断正确的测试图片数 / 测试图片总数
```
