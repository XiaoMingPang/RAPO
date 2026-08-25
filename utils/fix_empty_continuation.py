# [RAPO-FIX] 原地修复前缀数据中的「空续写」样本，其余样本保持字节级不变。
#   - 只针对 bad 中 assistant(续写) 为空的样本：把前缀末尾 1 个 token 挪到续写，
#     使续写非空，同时同步 good 的前缀（good/bad 共享同一前缀）。
#   - 不重新随机生成 k，因此 375 条正常样本的前缀 token 与原来完全一致，
#     可用于与旧结果对比。
# 用法：
#   python3 utils/fix_empty_continuation.py \
#     gemma-2-2b-it \
#     dataset/prefix/gemma-2-2b-it-prefix-good.jsonl \
#     dataset/prefix/gemma-2-2b-it-prefix-bad.jsonl
import json
import sys
from transformers import AutoTokenizer


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dump(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def get_msg(item, role):
    for m in item["messages"]:
        if m["role"] == role:
            return m
    raise KeyError(f"role {role} not found in {item}")


def main():
    model_path = sys.argv[1]
    good_path = sys.argv[2]
    bad_path = sys.argv[3]

    tok = AutoTokenizer.from_pretrained(model_path)
    good = load(good_path)
    bad = load(bad_path)
    assert len(good) == len(bad), f"长度不一致: good={len(good)} bad={len(bad)}"

    fixed = 0
    skipped = 0
    for i in range(len(bad)):
        bad_ass = get_msg(bad[i], "assistant")["content"]
        if bad_ass.strip():
            continue  # 续写非空，不动

        bad_prefix = get_msg(bad[i], "prefix")["content"]
        good_prefix = get_msg(good[i], "prefix")["content"]
        assert bad_prefix == good_prefix, f"index {i} good/bad 前缀不一致"

        full_text = bad_prefix  # 空续写 => 前缀即完整有害响应
        all_ids = tok.encode(full_text, add_special_tokens=False)
        if len(all_ids) <= 1:
            print(f"  [skip] index {i}: 响应过短({len(all_ids)} token)，无法再切")
            skipped += 1
            continue

        new_prefix = tok.decode(all_ids[:-1], skip_special_tokens=False)
        new_cont = tok.decode(all_ids[-1:], skip_special_tokens=False)

        get_msg(bad[i], "prefix")["content"] = new_prefix
        get_msg(bad[i], "assistant")["content"] = new_cont
        get_msg(good[i], "prefix")["content"] = new_prefix  # 同步 good 前缀
        fixed += 1

    dump(good_path, good)
    dump(bad_path, bad)
    print(f"修复空续写 {fixed} 对；跳过 {skipped} 对；"
          f"其余 {len(bad) - fixed - skipped} 对保持原样")


if __name__ == "__main__":
    main()
