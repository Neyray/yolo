"""
v8 二阶段推理：YOLO（单类检测） + ResNet（三类分类）。

流程：
1. stage1 YOLO 在原图上检测所有"果子"框
2. 把每个框 crop 出来送进 stage2 分类器
3. 输出最终的"位置 + 成熟度"结果

可以直接和 v6_aug 的一阶段输出做对照。
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

CLS_NAMES = ["immature", "mature", "overripe"]
COLORS = {"immature": (0, 200, 0), "mature": (220, 60, 20), "overripe": (40, 40, 40)}

VAL_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    classes = state.get("classes", CLS_NAMES)
    return model, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="/home/jerico/projects/fruit_detect/models/train_v8_stage1/weights/best.pt")
    ap.add_argument("--classifier", default="/home/jerico/projects/fruit_detect/models/train_v8_stage2/best.pt")
    ap.add_argument("--source", required=True, help="单张图或目录")
    ap.add_argument("--out", default="/home/jerico/projects/fruit_detect/models/predict_v8")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    detector = YOLO(args.detector)
    cls_model, cls_names = load_classifier(args.classifier, device)

    src = Path(args.source)
    paths = [src] if src.is_file() else sorted(src.iterdir())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = Image.open(p).convert("RGB")
        # stage1：YOLO 单类检测
        res = detector.predict(str(p), conf=args.conf, device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        # stage2：对每个框 crop -> 分类
        results = []
        for (x1, y1, x2, y2) in boxes:
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            x = VAL_TF(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(cls_model(x), dim=1)[0].cpu().numpy()
            cls_id = int(probs.argmax())
            results.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "class": cls_names[cls_id],
                "prob": float(probs[cls_id]),
            })

        # 画出来
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        for r in results:
            color = COLORS.get(r["class"], (200, 200, 0))
            draw.rectangle(r["xyxy"], outline=color, width=3)
            draw.text((r["xyxy"][0], max(0, r["xyxy"][1] - 12)),
                      f"{r['class']} {r['prob']:.2f}", fill=color)
        vis.save(out_dir / p.name)
        print(f"{p.name}: {len(results)} 个目标")

    print(f"\n输出在：{out_dir}")


if __name__ == "__main__":
    main()
