import json
import argparse
import re
from rule_penalty import RulePenalty


def extract_pku_responses(item: dict):
    """从 PKU-SafeRLHF 格式提取两条 response"""
    out = []
    for i in [0, 1]:
        text = item.get(f"response_{i}", "").strip()
        if text:
            out.append({
                "text": text,
                "response_id": i,
                "is_safe": item.get(f"is_response_{i}_safe", True),
                "severity": item.get(f"response_{i}_severity_level", 0),
            })
    return out


def evaluate_and_write(item, content, rule_engine, f_out, source_tag, stats):
    """对单条响应评估并写入输出"""
    eval_result = rule_engine.evaluate(content)
    item["rule_annotation"] = {
        "total_penalty": eval_result["total_penalty"],
        "matched_rules": [r["name"] for r in eval_result["matches"]],
        "full_matches": eval_result["matches"],
    }
    item["teacher_label"] = eval_result["total_penalty"]
    item["label_source"] = source_tag
    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats["labeled"] += 1
    if eval_result["total_penalty"] < 0:
        stats["unsafe"] += 1


def process_dataset(input_path, output_path, rule_path):
    rule_engine = RulePenalty(rule_path)
    stats = {"total": 0, "labeled": 0, "skipped": 0, "unsafe": 0}

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        line_num = 0
        for line in f_in:
            line_num += 1
            stats["total"] += 1
            try:
                item = json.loads(line.strip())

                # === 格式 1: OpenAI chat messages ===
                if "messages" in item:
                    content = None
                    for msg in item["messages"]:
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "").strip()
                            break
                    if content:
                        evaluate_and_write(item, content, rule_engine, f_out, "messages", stats)
                    else:
                        stats["skipped"] += 1

                # === 格式 2: 单 response 字段 ===
                elif "response" in item and "response_0" not in item:
                    content = (item["response"] or "").strip()
                    if content:
                        evaluate_and_write(item, content, rule_engine, f_out, "response", stats)
                    else:
                        stats["skipped"] += 1

                # === 格式 3: Alpaca (output 字段) ===
                elif "output" in item:
                    content = (item["output"] or "").strip()
                    if content:
                        evaluate_and_write(item, content, rule_engine, f_out, "alpaca", stats)
                    else:
                        stats["skipped"] += 1

                # === 格式 4: PKU-SafeRLHF (response_0 + response_1) ===
                elif "response_0" in item and "response_1" in item:
                    pku_responses = extract_pku_responses(item)
                    if not pku_responses:
                        stats["skipped"] += 1
                        continue
                    # 每条 response 单独写一行
                    for resp in pku_responses:
                        new_item = dict(item)
                        new_item["response"] = resp["text"]
                        new_item["pku_response_id"] = resp["response_id"]
                        new_item["pku_is_safe"] = resp["is_safe"]
                        new_item["pku_severity"] = resp["severity"]
                        evaluate_and_write(
                            new_item, resp["text"], rule_engine, f_out,
                            f"pku_response_{resp['response_id']}", stats
                        )

                # === 未知格式 ===
                else:
                    stats["skipped"] += 1
                    if stats["skipped"] <= 3:
                        print(f"跳过第{line_num}行：未识别格式，字段={list(item.keys())[:5]}")

            except json.JSONDecodeError:
                stats["skipped"] += 1
                if stats["skipped"] <= 3:
                    print(f"跳过第{line_num}行：JSON解析错误")
            except Exception as e:
                stats["skipped"] += 1
                if stats["skipped"] <= 3:
                    print(f"第{line_num}行处理失败：{str(e)}")

    print(f"\n📊 完成统计:")
    print(f"  总行数: {stats['total']}")
    print(f"  已标注: {stats['labeled']}")
    print(f"  跳过:   {stats['skipped']}")
    if stats["labeled"] > 0:
        print(f"  不安全 (penalty<0): {stats['unsafe']} ({stats['unsafe']/stats['labeled']*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rules", default="utils/safety_rules.json")
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.rules)