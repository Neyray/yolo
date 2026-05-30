"""
v8 stage2：在 stage1 检出的果子 crop 上做三类分类（immature / mature / overripe）。

用 torchvision ImageFolder 加载 data/labeled_v6_crops/{train,valid}/{immature,mature,overripe}/。
backbone 用 ImageNet 预训练 resnet18，最后一层换成 3 类。
训练完输出 best 权重到 models/train_v8_stage2/best.pt。

为什么这么做：v6_aug 的混淆矩阵显示 mature/overripe 边界是最弱环节，单一阶段
YOLO 既要学定位又要学细粒度分类，两个任务互相干扰。二阶段把定位丢给 stage1
（已经做得很稳），分类丢给 stage2，stage2 只看 crop 出来的果子局部，能聚焦
颜色/纹理/斑点这些细粒度特征。
"""
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

ROOT = Path("/home/jerico/projects/fruit_detect")
CROPS = ROOT / "data" / "labeled_v6_crops"
OUT = ROOT / "models" / "train_v8_stage2"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
BATCH = 64
LR = 1e-3
IMG = 128  # 果子 crop 不大，128 足够；想更精细可改 224

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(CROPS / "train", transform=train_tf)
val_ds = datasets.ImageFolder(CROPS / "valid", transform=val_tf)

print(f"classes (顺序按字典序): {train_ds.classes}")
print(f"train: {len(train_ds)}  val: {len(val_ds)}")

# 类别不平衡：weighted sampler，把 overripe 上采样到和 mature 同概率
cls_counts = [0, 0, 0]
for _, y in train_ds.samples:
    cls_counts[y] += 1
print(f"训练集每类数量: {dict(zip(train_ds.classes, cls_counts))}")
weights = [1.0 / cls_counts[y] for _, y in train_ds.samples]
sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# 模型：resnet18 + 改最后一层
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(DEVICE)

# 训练
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)  # 邻类边界软化

best_acc = 0.0
history = []
for ep in range(EPOCHS):
    model.train()
    tr_loss, tr_correct, tr_n = 0, 0, 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * x.size(0)
        tr_correct += (out.argmax(1) == y).sum().item()
        tr_n += x.size(0)
    sched.step()

    model.eval()
    val_correct, val_n = 0, 0
    per_class_correct = [0, 0, 0]
    per_class_total = [0, 0, 0]
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            val_correct += (pred == y).sum().item()
            val_n += x.size(0)
            for c in range(3):
                m = y == c
                per_class_total[c] += m.sum().item()
                per_class_correct[c] += ((pred == y) & m).sum().item()

    val_acc = val_correct / max(val_n, 1)
    per_cls_acc = [c / max(t, 1) for c, t in zip(per_class_correct, per_class_total)]
    rec = {
        "epoch": ep + 1,
        "train_loss": tr_loss / tr_n,
        "train_acc": tr_correct / tr_n,
        "val_acc": val_acc,
        "per_class_acc": dict(zip(train_ds.classes, per_cls_acc)),
    }
    history.append(rec)
    print(f"ep {ep+1:02d} | train_loss={rec['train_loss']:.4f} train_acc={rec['train_acc']:.4f} | val_acc={val_acc:.4f} | per_cls={rec['per_class_acc']}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(), "classes": train_ds.classes}, OUT / "best.pt")

torch.save({"model": model.state_dict(), "classes": train_ds.classes}, OUT / "last.pt")
(OUT / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
print(f"\n最优验证集准确率：{best_acc:.4f}")
print(f"权重保存在：{OUT}")
