import json
import sys
from transformers import AutoTokenizer
import numpy as np

# [RAPO-FIX-EXTRA] 修复原脚本的 NameError（ids / first_k_tokens_len 未定义）
# 与越界问题（k 可能超过 bad_msg 的 token 数导致续写为空），
# 统一用 token-id 切分（prefix_ids + suffix_ids = all_ids，保证 round-trip）。
combined_k = np.random.randint(1, 257, size=400).tolist()
np.random.shuffle(combined_k)
combined_k = combined_k.tolist()

model_path = sys.argv[1]
input_good = sys.argv[2]
input_bad = sys.argv[3]
output_good = sys.argv[4]
output_bad = sys.argv[5]

good_data = []
bad_data = []

with open(input_good, 'r') as input_file:
    lines = input_file.readlines()
for line in lines:
    json_obj = json.loads(line.strip())
    good_data.append(json_obj)

with open(input_bad, 'r') as input_file:
    lines = input_file.readlines()
for line in lines:
    json_obj = json.loads(line.strip())
    bad_data.append(json_obj)

tokenizer = AutoTokenizer.from_pretrained(model_path)

good_ans = []
bad_ans = []

for i, item in enumerate(bad_data):
    bad_msg = item["messages"][1]["content"]
    user_msg = item["messages"][0]["content"]
    k = combined_k[i]

    all_ids = tokenizer.encode(bad_msg, add_special_tokens=False)
    # 越界保护：前缀最多切到 len(all_ids)-1，保证续写至少 1 个 token
    k = min(k, max(0, len(all_ids) - 1))
    prefix_ids = all_ids[:k]
    suffix_ids = all_ids[k:]
    prefix_str = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=False)

    bad_ans.append({
        "messages": [
            {
                "role": "user",
                "content": user_msg
            },
            {
                "role": "prefix",
                "content": prefix_str
            },
            {
                "role": "assistant",
                "content": suffix_str
            }
        ]
    })
    good_ans.append({
        "messages": [
            {
                "role": "user",
                "content": user_msg
            },
            {
                "role": "prefix",
                "content": prefix_str
            },
            {
                "role": "assistant",
                "content": good_data[i]["messages"][1]["content"]
            }
        ]
    })

with open(output_good, 'w') as output_file:
    for item in good_ans:
        output_file.write(json.dumps(item) + '\n')

with open(output_bad, 'w') as output_file:
    for item in bad_ans:
        output_file.write(json.dumps(item) + '\n')
