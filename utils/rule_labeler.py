import json
import argparse
from rule_reward import RuleReward

def process_dataset(input_path, output_path, rule_path):
    """为数据集添加规则匹配标签（兼容messages/response两种格式）"""
    rule_engine = RuleReward(rule_path)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        line_num = 0
        for line in f_in:
            line_num += 1
            try:
                item = json.loads(line.strip())
                content = None
                
                # 处理1：多轮对话格式（messages字段）
                if "messages" in item:
                    for msg in item["messages"]:
                        if msg["role"] == "assistant":
                            content = msg["content"]
                            break
                # 处理2：单响应格式（response字段）
                elif "response" in item:
                    content = item["response"]
                # 处理3：无有效内容，跳过
                if content is None:
                    print(f"跳过第{line_num}行：无assistant/response内容")
                    continue
                
                # 规则评估
                eval_result = rule_engine.evaluate(content)
                
                # 添加标签（兼容训练脚本的teacher_label字段）
                item["rule_annotation"] = {
                    "total_penalty": eval_result["total_penalty"],
                    "matched_rules": [r["name"] for r in eval_result["matches"]],
                    "full_matches": eval_result["matches"]
                }
                # 新增：添加训练脚本需要的teacher_label字段
                item["teacher_label"] = eval_result["total_penalty"]
                
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"跳过第{line_num}行：JSON解析错误")
            except Exception as e:
                print(f"第{line_num}行处理失败：{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始数据集路径（JSONL格式）")
    parser.add_argument("--output", required=True, help="带规则标签的输出路径")
    parser.add_argument("--rules", default="utils/safety_rules.json", help="规则文件路径")
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.rules)