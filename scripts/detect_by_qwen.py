from openai import OpenAI
import base64
import sys
import os

# 配置区
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


def detect_fruit(image_path):
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
        return

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    ext = image_path.split(".")[-1].lower()
    media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
    media_type = media_type_map.get(ext, "image/jpeg")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen-vl-plus",  
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{img_data}"
                    }
                },
                {
                    "type": "text",
                    "text": """请分析这张树的图片，回答以下问题：
1. 这是什么种类的树？
2. 树上有没有果子？
3. 如果有果子，大概有几个（给出估计数量）？
4. 果子的成熟程度如何？

请按以下格式回答：
树种：xxx
有无果子：有/没有
果子数量：x个（没有则填0）
成熟程度：xxx
停车建议：适合停车/不适合停车"""
                }
            ]
        }]
    )

    result = response.choices[0].message.content
    print("=" * 40)
    print(f"图片：{image_path}")
    print("=" * 40)
    print(result)
    print("=" * 40)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式：python detect_by_qwen.py ./data/raw/tree.jpg")
    else:
        detect_fruit(sys.argv[1])