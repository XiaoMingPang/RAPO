# ============================================================
# [RAPO-FIX3+4] 直接用 prefill 好/坏对训练「惩罚分类器」
#   - 替代 train_rule_distiller.py 的规则蒸馏路径，作为 RAPO 的
#     penalty_model（训练后用 --rule_model_path 指向本脚本输出的
#     final_model 即可）。
#   - 数据：prefix-good.jsonl（label=0，安全拒绝续写）+
#            prefix-bad.jsonl（label=1，有害续写）
#   - 每条样本格式与 loss.py 的 _format_penalty_texts 完全一致：
#       "Question: {user}\nResponse: {continuation}"
#     续写不含攻击者前缀（messages[2] 就是续写），修复了规则老师在
#     前缀攻击分布上失明/反向的根因。
#   - 训练：AutoModelForSequenceClassification(num_labels=1) +
#     BCEWithLogitsLoss。
#   - 打分时 loss.py 对 logit 做 -sigmoid：安全≈0(risk≈0)，有害≈-1(risk≈1)。
# ============================================================
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _extract_user_assistant(item):
    """兼容两种格式：
      - prefix 数据：[user, prefix, assistant]（assistant 即续写，不含前缀）
      - 普通数据：[user, assistant]
    返回 (user_text, assistant_text)，缺任一字段返回 None。
    """
    user_text = None
    assistant_text = None
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            user_text = msg.get("content")
        elif msg.get("role") == "assistant":
            assistant_text = msg.get("content")
    if not user_text or not assistant_text:
        return None
    return user_text, assistant_text


def build_sample(item, label):
    parsed = _extract_user_assistant(item)
    if parsed is None:
        return None
    user_text, assistant_text = parsed
    # [RAPO-FIX2] 格式必须与 loss.py 的 _format_penalty_texts 保持一致
    text = f"Question: {user_text}\nResponse: {assistant_text}"
    return {"text": text, "label": float(label)}


def load_samples(paths_and_labels):
    samples = []
    for path, label in paths_and_labels:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sample = build_sample(item, label)
                if sample is not None:
                    samples.append(sample)
    return samples


def load_raw_items(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


@torch.no_grad()
def report_risk_gap(good_path, bad_path, model, tokenizer,
                    temperature=1.0, max_length=512, max_pairs=1000):
    """[RAPO-FIX-Q2] 自适应报告：在 prefix good/bad 对上计算 risk_gap 分布，
    验证「动态风险边际」是否真的随样本变化（供论文作动态性证据）。"""
    model.eval()
    good_items = load_raw_items(good_path)
    bad_items = load_raw_items(bad_path)
    n = min(len(good_items), len(bad_items), max_pairs)
    if n == 0:
        print("  [WARN] 无足够 prefix 对，跳过 risk_gap 报告")
        return
    gaps = []
    for i in range(n):
        g = build_sample(good_items[i], 0.0)
        b = build_sample(bad_items[i], 1.0)
        if g is None or b is None:
            continue
        texts = [g["text"], b["text"]]
        enc = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True,
            max_length=max_length,
        ).to(model.device)
        logits = model(**enc).logits.squeeze(-1).float()
        penalties = -torch.sigmoid(logits / temperature)
        risks = -penalties
        gap = (risks[1] - risks[0]).clamp(0.0, 1.0).item()
        gaps.append(gap)
    if not gaps:
        print("  [WARN] risk_gap 报告无有效样本")
        return
    arr = np.array(gaps)
    print(f"[ADAPTIVITY] risk_gap over {len(arr)} prefix pairs (T={temperature}):")
    print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  zero-ish(<0.05)={100*(arr < 0.05).mean():.1f}%  "
          f"near-1(>0.95)={100*(arr > 0.95).mean():.1f}%")
    hist, _ = np.histogram(arr, bins=[0, 0.25, 0.5, 0.75, 1.0, 1.01])
    print(f"  hist[0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]: {hist.tolist()}")
    if (arr > 0.95).mean() > 0.8:
        print("  [!] risk_gap 高度饱和（>80% 样本接近 1）→ margin 近似常数。")
        print("      建议 --score_temperature 1.5~2.0 或 --extra_good/--extra_bad 混多样数据，增强动态性。")


class PenaltyDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(s["label"], dtype=torch.float32),
        }


class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # [RAPO-FIX] labels 可能是 [N]（1-D），不能 squeeze(-1)；统一 reshape 到 1-D 更稳
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    preds = (logits > 0.0).astype(np.int64)
    labels_int = labels.astype(np.int64)
    acc = accuracy_score(labels_int, preds)
    good_logits = logits[labels_int == 0]
    bad_logits = logits[labels_int == 1]
    return {
        "accuracy": float(acc),
        "good_logit_mean": float(good_logits.mean()) if len(good_logits) else float("nan"),
        "bad_logit_mean": float(bad_logits.mean()) if len(bad_logits) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="训练 RAPO 直接惩罚分类器（替代规则蒸馏）")
    parser.add_argument("--good_data", required=True,
                        help="prefix-good.jsonl（label=0，安全拒绝续写）")
    parser.add_argument("--bad_data", required=True,
                        help="prefix-bad.jsonl（label=1，有害续写）")
    parser.add_argument("--base_model", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra_good", nargs="*", default=None,
                        help="可选：普通 good 数据（增强规模 / 校准，messages=[user, assistant]）")
    parser.add_argument("--extra_bad", nargs="*", default=None,
                        help="可选：普通 bad 数据")
    parser.add_argument("--score_temperature", type=float, default=1.0,
                        help="risk_gap 报告用温度（与 train.py --score_temperature 一致）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths_and_labels = [(args.good_data, 0.0), (args.bad_data, 1.0)]
    if args.extra_good:
        paths_and_labels.extend([(p, 0.0) for p in args.extra_good])
    if args.extra_bad:
        paths_and_labels.extend([(p, 1.0) for p in args.extra_bad])

    samples = load_samples(paths_and_labels)
    n_good = sum(1 for s in samples if s["label"] == 0.0)
    n_bad = sum(1 for s in samples if s["label"] == 1.0)
    print(f"Loaded {len(samples)} samples (good={n_good}, bad={n_bad})")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = PenaltyDataset(train_samples, tokenizer, args.max_length)
    val_ds = PenaltyDataset(val_samples, tokenizer, args.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
    )

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("===== Train Direct Penalty Classifier =====")
    trainer.train()

    final_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Classifier saved to {final_dir}")
    print("下一步：train.py --rule_model_path 指向该 final_model 目录")

    # [RAPO-FIX-Q2] 动态性自检：看 risk_gap 分布是否显著变化、是否饱和
    report_risk_gap(
        args.good_data, args.bad_data, trainer.model, tokenizer,
        temperature=args.score_temperature, max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
