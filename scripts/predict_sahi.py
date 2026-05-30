"""
SAHI 切片推理。在不重新训练的前提下，看小目标 Recall 能不能涨。

逻辑：把每张原图切成若干 overlap 的小块，分别送进 YOLO，再把所有小块的预测
合并回原图坐标系（NMS 去重）。原理是把"小目标"放大成"中等目标"，避开 YOLO
在低分辨率特征图上对小目标的特征塌陷问题。

依赖：pip install sahi
"""
import argparse
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/jerico/projects/fruit_detect/models/train_v6_aug/weights/best.pt")
    ap.add_argument("--source", default="/home/jerico/projects/fruit_detect/data/labeled_v6/valid/images")
    ap.add_argument("--out", default="/home/jerico/projects/fruit_detect/models/predict_sahi_v6_aug")
    ap.add_argument("--slice", type=int, default=640, help="切片边长")
    ap.add_argument("--overlap", type=float, default=0.2, help="切片重叠比例")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 批量预测整个目录，结果会写到 out 下
    predict(
        detection_model=model,
        source=str(src),
        project=str(out.parent),
        name=out.name,
        slice_height=args.slice,
        slice_width=args.slice,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        export_visual=True,
        no_standard_prediction=False,  # 同时跑一遍标准推理用于对比
        novisual=False,
    )

    print(f"\nSAHI 推理完成。可视化结果在 {out}")
    print("对照实验：")
    print(f"  原生推理：yolo predict model={args.weights} source={src} save=True")
    print(f"  SAHI 推理：本脚本输出（{out}）")


if __name__ == "__main__":
    main()
