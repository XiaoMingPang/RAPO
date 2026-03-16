import json
import random

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

random.shuffle(data)

n = len(data)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_data = data[:n_train]
val_data   = data[n_train:n_train+n_val]
test_data  = data[n_train+n_val:]

def dump(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

dump(OUT_TRAIN, train_data)
dump(OUT_VAL, val_data)
dump(OUT_TEST, test_data)

print(f"Train: {len(train_data)}")
print(f"Val:   {len(val_data)}")
print(f"Test:  {len(test_data)}")