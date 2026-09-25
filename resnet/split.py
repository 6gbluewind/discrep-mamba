import json
import os

def split_brats_json(input_file):
    # 加载原始 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 定义模态名称及其在 images 列表中的索引
    modalities = {
        "t1": 0,
        "t1ce": 1,
        "t2": 2,
        "flair": 3
    }

    # 遍历每个模态并生成对应的 JSON 内容
    for mod_name, index in modalities.items():
        new_data = {"train": [], "val": []}

        # 处理训练集
        for item in data.get("train", []):
            new_item = {
                "images": [item["images"][index]], # 保持列表格式，但只包含一个路径
                "label": item["label"]
            }
            new_data["train"].append(new_item)

        # 处理验证集
        for item in data.get("val", []):
            new_item = {
                "images": [item["images"][index]],
                "label": item["label"]
            }
            new_data["val"].append(new_item)

        # 保存为新的 JSON 文件
        output_filename = f"brats2018_{mod_name}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成: {output_filename}")

if __name__ == "__main__":
    # 请确保 input.json 与此脚本在同一目录下，或者提供绝对路径
    input_json_path = "brats2018.json" 
    
    if os.path.exists(input_json_path):
        split_brats_json(input_json_path)
    else:
        print(f"错误: 找不到文件 {input_json_path}")