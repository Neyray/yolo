# fruit_detect

基于 YOLO11 的水果成熟度检测项目，识别三类目标：`immature_fruit`、`mature_fruit`、`overripe_fruit`。

## 目录结构

```
fruit_detect/
├── data/
│   ├── labeled/        # v1 数据集
│   ├── labeled_v2/     # v2 数据集
│   ├── labeled_v4/     # v4 数据集（当前训练）
│   └── raw/
├── models/             # 训练输出（权重、日志、曲线图）
│   ├── train_v1/
│   ├── train_v2/
│   └── train_v4/       # 训练后生成
├── scripts/
│   ├── train.py        # v1 训练脚本
│   ├── train_v2.py     # v2 训练脚本
│   ├── train_v4.py     # v4 训练脚本
│   ├── detect_by_yolo.py
│   └── detect_by_qwen.py
└── results/
```

## 训练 v4 数据

### 1. 环境准备

确保已激活含 `ultralytics` 的 conda 环境：

```bash
conda activate <你的环境名>
pip install ultralytics
```

确认 GPU 可用（脚本默认 `device=0`）：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. 数据集检查

`data/labeled_v4/data.yaml` 应包含：

- `train: ../train/images`
- `val: ../valid/images`
- `test: ../test/images`
- `nc: 3`
- `names: ['immature_fruit', 'mature_fruit', 'overripe_fruit']`

### 3. 启动训练

```bash
cd /home/jerico/projects/fruit_detect
python scripts/train_v4.py
```

后台训练（断开终端不中断）：

```bash
cd /home/jerico/projects/fruit_detect
nohup python scripts/train_v4.py > models/train_v4.log 2>&1 &
tail -f models/train_v4.log
```

### 4. 训练参数

`scripts/train_v4.py` 默认参数：

| 参数 | 值 |
|------|----|
| 预训练权重 | `yolo11n.pt` |
| epochs | 150 |
| imgsz | 640 |
| device | 0 (GPU) |
| patience | 30（早停） |
| project | `models/` |
| name | `train_v4` |

如需调整，直接编辑 `scripts/train_v4.py`。

### 5. 训练产物

训练完成后，结果保存在 `models/train_v4/`：

- `weights/best.pt` —— 验证集上最优权重
- `weights/last.pt` —— 最后一个 epoch 的权重
- `results.png`、`confusion_matrix.png` 等可视化结果
- `results.csv` —— 每个 epoch 的指标日志

### 6. 推理验证

```bash
yolo predict model=models/train_v4/weights/best.pt source=data/labeled_v4/test/images
```
