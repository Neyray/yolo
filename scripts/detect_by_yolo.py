from ultralytics import YOLO
import sys
import os

# ===== 配置区 =====
# 训练好之后把 best.pt 放到 models/ 文件夹
# 没训练之前可以先用 yolo11n.pt 测试流程
MODEL_PATH = "./models/train_v5/weights/best.pt"
#脚本里 RESULTS_DIR = "./results" 但YOLO内部会自动加一层路径，导致实际保存到了 runs/detect/results/predict
RESULTS_DIR = "/home/jerico/projects/fruit_detect/results"
# ==================

def detect_fruit(image_path):
    """
    输入：图片路径
    输出：检测结果图（保存到results/），果子数量
    """
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型 {MODEL_PATH}")
        print("提示：训练完成后把 best.pt 放到 models/ 文件夹")
        print("提示：或者先用 yolo11n.pt 测试流程，把 MODEL_PATH 改成 yolo11n.pt")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型
    model = YOLO(MODEL_PATH)

    # 推理
    results = model(
        image_path,
        save=True,           # 保存结果图
        project=RESULTS_DIR, # 保存位置
        name="predict",
        exist_ok=True
    )

    # 统计果子数量
    fruit_count = len(results[0].boxes)

    print("=" * 40)
    print(f"图片：{image_path}")
    print(f"检测到果子数量：{fruit_count} 个")

    if fruit_count == 0:
        print("结论：没有果子 ✅ 适合停车")
    elif fruit_count <= 5:
        print(f"结论：有少量果子({fruit_count}个) ⚠️ 谨慎停车")
    else:
        print(f"结论：果子较多({fruit_count}个) ❌ 不建议停车")

    print(f"结果图已保存到：{RESULTS_DIR}/predict/")
    print("=" * 40)

    return fruit_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式：python detect_by_yolo.py ./data/raw/tree.jpg")
    else:
        detect_fruit(sys.argv[1])
