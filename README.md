# 果实检测：YOLO 果实状态识别与停车风险评估

## 研究方向

当前方向：

> 面向停车安全的仰视果实检测：先用 YOLO 检测果实状态/类型，再评估车是否适合停在树下。

后续不要继续把项目只当成"数果子"。旧的单类别模型可以保留作为基线模型，但主模型应改成多类别检测。

## 推荐类别

Roboflow 中建议使用这 4 个类别：

```yaml
nc: 4
names:
  - immature_fruit
  - mature_fruit
  - overripe_fruit
  - fruit_cluster
```

类别含义：

```text
immature_fruit: 未成熟果实
mature_fruit: 成熟果实
overripe_fruit: 过熟、破损、干枯或明显高风险果实
fruit_cluster: 果实密集成簇，难以逐个框出时使用
```

标注规则：

1. 单个果实边界清楚时，尽量框单个果实。
2. 多个果实重叠、边界难以区分时，整簇标为 `fruit_cluster`。
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

旧的单类别数据集如果继续保留，只建议作为基线对比使用。

## 安装依赖

```bash
cd /home/jerico/projects/fruit_detect
pip install -r requirements.txt
```

## 训练

默认训练命令：

```bash
cd /home/jerico/projects/fruit_detect
python scripts/train.py
```

常用参数：

```bash
python scripts/train.py --weights yolo11s.pt --epochs 150 --batch 8 --name fruit_risk_yolo11s
python scripts/train.py --data /path/to/data.yaml --device cpu
```

默认输出模型位置：

```text
models/fruit_risk_yolo11n/weights/best.pt
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
fruit_cluster/dense_fruit_cluster: 4.0
```

停车建议规则：

```text
高风险：高风险果实 >= 2，或果簇 >= 2，或 risk_score >= 12
中风险：高风险果实 >= 1，或果簇 >= 1，或成熟果实 >= 3，或 risk_score >= 6
低风险：没有检测到果实，或主要是未成熟/低风险果实
```

这里故意使用可解释的规则，方便写进论文或汇报。

## Qwen-VL 对比方法

Qwen-VL 适合作为辅助分析或对比基线，不建议作为唯一方法。

使用前先设置 API Key：

```bash
export DASHSCOPE_API_KEY='your_key_here'
python scripts/detect_by_qwen.py data/raw/01.png
```

如果 API Key 曾经写进代码、截图或发给别人，请去控制台撤销旧 Key，并重新生成。

## 推荐实验设计

1. 基线实验：旧单类别 YOLO，只统计果实数量。
2. 主方法实验：多类别 YOLO，识别果实成熟度/状态。
3. 风险评估实验：YOLO 检测结果 + 规则化风险评分。
4. 可选对比实验：Qwen-VL 视觉分析。

YOLO 指标建议汇报：

```text
Precision
Recall
mAP50
mAP50-95
per-class recall
confusion matrix
```

停车风险指标可以自定义：

```text
risk_accuracy = 风险判断正确的测试图片数 / 测试图片总数
```
