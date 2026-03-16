# merge_labeled_data.py
import json
import argparse

def merge_labeled_datasets(input_paths, output_path):
    """合并多个带规则标注的JSONL文件"""
    merged_data = []
    seen_texts = set()  # 去重（可选）
    
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 提取assistant内容用于去重
                    content = None
                    if "messages" in item:
                        for msg in item["messages"]:
                            if msg["role"] == "assistant":
                                content = msg["content"]
                                break
                    elif "response" in item:
                        content = item["response"]
                    
                    # 去重（可选，避免重复样本）
                    if content and content not in seen_texts:
                        seen_texts.add(content)
                        merged_data.append(item)
                except Exception as e:
                    print(f"跳过无效行：{str(e)}")
    
    # 保存合并后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成！总样本数：{len(merged_data)}")
    print(f"保存路径：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="待合并的标注文件路径列表")
    parser.add_argument("--output", required=True, help="合并后的输出路径")
    args = parser.parse_args()
    
    merge_labeled_datasets(args.inputs, args.output)