import argparse
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import transformers
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Any
from scipy.stats import spearmanr

from rule_penalty import RulePenalty


@dataclass
class CustomDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        teacher_labels = [feature.pop("teacher_label") for feature in features]
        batch = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of
        )(features)
        batch["teacher_label"] = torch.stack(teacher_labels)
        return batch


class DistilledRulePenaltyDataset(Dataset):
    def __init__(self, data_path, tokenizer, rule_path, max_length: int = 512,
                 oversample_unsafe: int = 1, oversample_seed: int = 42):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_length = max_length
        self.teacher = RulePenalty(rule_path)

        with open(data_path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    assistant_text = None
                    if "messages" in item:
                        for msg in item.get("messages", []):
                            if msg.get("role") == "assistant":
                                assistant_text = msg.get("content", "").strip()
                                break
                    elif "response" in item:
                        assistant_text = item.get("response", "").strip()

                    if not assistant_text:
                        print(f"[WARN] line {line_id} skipped: no assistant content")
                        continue
                    teacher_res = self.teacher.evaluate(assistant_text)
                    teacher_label = float(teacher_res["total_penalty"])
                    if (
                        "rule_annotation" in item
                        and "total_penalty" in item["rule_annotation"]
                    ):
                        student_label = float(item["rule_annotation"]["total_penalty"])
                    else:
                        student_label = teacher_label
                    self.samples.append({
                        "text": assistant_text,
                        "student_label": student_label,
                        "teacher_label": teacher_label,
                    })
                except Exception as e:
                    print(f"[WARN] line {line_id} skipped: {e}")

        if oversample_unsafe > 1:
            random.seed(oversample_seed)
            unsafe = [s for s in self.samples if s["teacher_label"] < 0]
            safe = [s for s in self.samples if s["teacher_label"] >= 0]
            print(f"[Oversample] Before: safe={len(safe)}, unsafe={len(unsafe)}")
            self.samples = safe + unsafe * oversample_unsafe
            random.shuffle(self.samples)
            n_unsafe_new = len(unsafe) * oversample_unsafe
            print(f"[Oversample] After:  safe={len(safe)}, unsafe={n_unsafe_new}")
            print(f"[Oversample] Ratio:  safe={len(safe)/len(self.samples)*100:.1f}%, "
                  f"unsafe={n_unsafe_new/len(self.samples)*100:.1f}%")
        else:
            n_unsafe = sum(1 for s in self.samples if s["teacher_label"] < 0)
            n_safe = len(self.samples) - n_unsafe
            print(f"[Dataset] safe={n_safe}, unsafe={n_unsafe} "
                  f"(ratio: {n_safe/len(self.samples)*100:.1f}/{n_unsafe/len(self.samples)*100:.1f}%)")

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
            "labels": torch.tensor(s["student_label"], dtype=torch.float32),
            "teacher_label": torch.tensor(s["teacher_label"], dtype=torch.float32),
        }


class DistillationTrainer(Trainer):
    def __init__(self, *args, warmup_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps

    def pairwise_ranking_loss(self, preds, teacher_scores, margin=0.0, eps=0.05):
        """✅ 修复: 去掉 'loss +' 前缀, 防止指数爆炸"""
        loss = 0.0
        cnt = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                delta = teacher_scores[i] - teacher_scores[j]
                if torch.abs(delta) < eps:
                    continue
                if delta < 0:
                    loss += torch.relu(margin + preds[i] - preds[j])  # ✅ FIX
                else:
                    loss += torch.relu(margin + preds[j] - preds[i])  # ✅ FIX
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=preds.device)
        return loss / cnt

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_target = inputs.pop("labels")
        teacher_target = inputs.pop("teacher_label")
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1)

        # 配合过采样 3x, 数据已 50/50 平衡, 用原版 k=0.1 即可
        weight = torch.abs(teacher_target) + 0.1

        hard_loss = F.smooth_l1_loss(preds, student_target)
        distill_loss = (weight * (preds - teacher_target) ** 2).mean()
        rank_loss = self.pairwise_ranking_loss(preds, teacher_target)

        step = max(1, self.state.global_step)
        alpha = min(1.0, step / self.warmup_steps)
        rank_weight = 0.3 * alpha
        total_loss = (
            alpha * hard_loss
            + (1.0 - alpha) * distill_loss
            + rank_weight * rank_loss
        )
        return (total_loss, outputs) if return_outputs else total_loss


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    labels = labels.squeeze()
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    spear = spearmanr(preds, labels).correlation
    return {
        "mse": float(mse),
        "mae": float(mae),
        "spearman": float(spear),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="distilbert-base-uncased")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--rule_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--oversample_unsafe", type=int, default=3,
                        help="不安全样本复制倍数")
    args = parser.parse_args()
    print(f"Transformers version: {transformers.__version__}")
    print(f"Using base model: {args.base_model}")
    print(f"Training data: {args.train_data}")
    print(f"Oversample unsafe: {args.oversample_unsafe}x")
    print(f"Max sequence length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        problem_type="regression",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = DistilledRulePenaltyDataset(
        args.train_data, tokenizer, args.rule_path,
        max_length=args.max_length,
        oversample_unsafe=args.oversample_unsafe,
    )
    val_ds = DistilledRulePenaltyDataset(
        args.val_data, tokenizer, args.rule_path,
        max_length=args.max_length,
        oversample_unsafe=1,
    )
    print("Dataset loaded:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")

    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

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
        metric_for_best_model="spearman",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        remove_unused_columns=False,
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        warmup_steps=args.warmup_steps,
    )
    print("===== Start Rule Reward Distillation Training =====")
    trainer.train()
    final_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining completed!")
    print(f"Model saved to {final_dir}")
    print("NOTE: Test set is reserved for post-hoc rule fidelity evaluation.")


if __name__ == "__main__":
    main()