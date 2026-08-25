import sys
import json


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def key(item):
    d = {x["role"]: x["content"] for x in item["messages"]}
    return (d.get("user", ""), d.get("prefix", ""))


def main():
    good = load(sys.argv[1])
    bad = load(sys.argv[2])
    n = min(len(good), len(bad))
    mismatch = 0
    for i in range(n):
        if key(good[i]) != key(bad[i]):
            mismatch += 1
            if mismatch <= 5:
                print(f"  index {i}: user/prefix 不一致")
    print(f"good={len(good)}  bad={len(bad)}  min={n}  对齐={n - mismatch}  错位={mismatch}")
    if len(good) != len(bad):
        extra = "good" if len(good) > len(bad) else "bad"
        print(f"[!] 长度不一致：多了 {abs(len(good) - len(bad))} 条 {extra}")
        print("    -> 建议用修复后的 create_prefix_dataset.py 重新生成，保证 good/bad 对齐且等长")
    elif mismatch == 0:
        print("OK：good/bad 逐条对齐")


if __name__ == "__main__":
    main()
