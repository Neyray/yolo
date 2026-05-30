"""
为 v8 二阶段方法准备数据。

输入：data/labeled_v6/{train,valid,test}/{images,labels}

输出：
1. data/labeled_v6_single/  —— 单类（fruit）YOLO 数据集，给 v8 stage1 检测用
   - 结构和 labeled_v6 一样，images 是软链接，labels 把每行的 class_id 全改成 0
   - data.yaml: nc=1, names=['fruit']

2. data/labeled_v6_crops/{train,valid,test}/{immature,mature,overripe}/*.jpg
   —— 按 ground truth 框裁出的果子小图，给 v8 stage2 分类用
   - 每张原图按 bbox 裁出多张子图，按真实类别归到对应目录
   - 用 ImageFolder 直接加载训练
"""
import shutil
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path("/home/jerico/projects/fruit_detect")
SRC = ROOT / "data" / "labeled_v6"
DST_SINGLE = ROOT / "data" / "labeled_v6_single"
DST_CROPS = ROOT / "data" / "labeled_v6_crops"

NAMES = ["immature_fruit", "mature_fruit", "overripe_fruit"]
SHORT = ["immature", "mature", "overripe"]
SPLITS = ["train", "valid", "test"]
PAD = 0.05  # 裁切时往外多扩 5% 边界，避免果子被切掉


def prepare_single_class():
    """复制 labels，把 class_id 全改成 0；images 用软链接。"""
    print(f"[stage1] 生成单类数据集 -> {DST_SINGLE}")
    for split in SPLITS:
        img_src = SRC / split / "images"
        lbl_src = SRC / split / "labels"
        img_dst = DST_SINGLE / split / "images"
        lbl_dst = DST_SINGLE / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in img_src.iterdir():
            link = img_dst / img.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(img.resolve())

        for lbl in lbl_src.iterdir():
            new = []
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                parts[0] = "0"
                new.append(" ".join(parts))
            (lbl_dst / lbl.name).write_text("\n".join(new) + "\n")

        print(f"  {split}: {len(list(img_dst.iterdir()))} 张图")

    yaml.safe_dump(
        {
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "nc": 1,
            "names": ["fruit"],
        },
        (DST_SINGLE / "data.yaml").open("w"),
        sort_keys=False,
        allow_unicode=True,
    )
    print(f"  写出 {DST_SINGLE / 'data.yaml'}")


def crop_one(img_path: Path, lbl_path: Path, out_root: Path):
    """对一张图，按 labels 裁出所有 bbox 存到 out_root/<class>/<img>_<i>.jpg。"""
    if not lbl_path.exists():
        return 0
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    n = 0
    for i, line in enumerate(lbl_path.read_text().splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        # YOLO 归一化坐标 -> 像素坐标
        x1 = (cx - w / 2 - PAD * w) * W
        y1 = (cy - h / 2 - PAD * h) * H
        x2 = (cx + w / 2 + PAD * w) * W
        y2 = (cy + h / 2 + PAD * h) * H
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue  # 太小的框跳过，避免分类器学到一堆噪声
        crop = img.crop((x1, y1, x2, y2))
        out_dir = out_root / SHORT[cls]
        out_dir.mkdir(parents=True, exist_ok=True)
        crop.save(out_dir / f"{img_path.stem}_{i}.jpg", quality=90)
        n += 1
    return n


def prepare_crops():
    """从 GT 框裁出小图，按 split 和类别归档，给分类器用。"""
    print(f"[stage2] 裁切果实小图 -> {DST_CROPS}")
    if DST_CROPS.exists():
        shutil.rmtree(DST_CROPS)
    counts = {s: {c: 0 for c in SHORT} for s in SPLITS}
    for split in SPLITS:
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        out_root = DST_CROPS / split
        for img in img_dir.iterdir():
            lbl = lbl_dir / (img.stem + ".txt")
            crop_one(img, lbl, out_root)
        # 统计
        for c in SHORT:
            d = out_root / c
            counts[split][c] = len(list(d.iterdir())) if d.exists() else 0
        print(f"  {split}: {counts[split]}")

    return counts


if __name__ == "__main__":
    prepare_single_class()
    crops = prepare_crops()
    print("\n完成。")
    print("下一步：")
    print("  python scripts/train_v8_stage1.py    # 单类 YOLO（fruit）")
    print("  python scripts/train_v8_stage2.py    # 三类分类器（resnet18）")
