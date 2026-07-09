import json
import random
from collections import defaultdict

INPUT = "./dataset/train/merged_rule_labeled.jsonl"
OUT_TRAIN = "./dataset/train/rule_train.jsonl"
OUT_VAL   = "./dataset/train/rule_val.jsonl"
OUT_TEST  = "./dataset/train/rule_test.jsonl"

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

random.seed(SEED)

with open(INPUT, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ✅ 按 teacher_label 桶分层
buckets = defaultdict(list)
for item in data:
    lbl = round(item.get("teacher_label", 0), 3)  # 按 label 桶分
    buckets[lbl].append(item)

train_data, val_data, test_data = [], [], []
for lbl, items in buckets.items():
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_data.extend(items[:n_train])
    val_data.extend(items[n_train:n_train + n_val])
    test_data.extend(items[n_train + n_val:])

# 再打乱一次，避免同一 label 集中
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

def dump(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

dump(OUT_TRAIN, train_data)
dump(OUT_VAL, val_data)
dump(OUT_TEST, test_data)

# 验证分布
from collections import Counter
for name, d in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
    labels = [round(x.get("teacher_label", 0), 3) for x in d]
    n_unsafe = sum(1 for l in labels if l < 0)
    print(f"{name}: {len(d)} 条, 不安全 {n_unsafe} ({n_unsafe/len(d)*100:.1f}%)")