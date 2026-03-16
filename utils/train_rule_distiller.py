import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
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
# ===========================================================
# 0. Teacher：直接复用归一化 RuleReward（唯一真源）
# ===========================================================
from rule_reward import RuleReward
# ===========================================================
# 自定义DataCollator（保留teacher_label字段）
# ===========================================================
@dataclass
class CustomDataCollatorWithPadding:
    """
    自定义DataCollator，保留teacher_label等自定义字段
    """
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离出非tokenizer字段
        teacher_labels = [feature.pop("teacher_label") for feature in features]
        # 使用标准的DataCollator处理tokenizer相关字段
        batch = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of
        )(features)
        
        # 把teacher_label加回batch中
        batch["teacher_label"] = torch.stack(teacher_labels)        
        return batch
# ===========================================================
# 1. 蒸馏数据集（✔ labels / teacher_label 分离）
# ===========================================================
class DistilledRuleRewardDataset(Dataset):
    """
    Dataset for rule reward distillation.
    Each sample:
        text
        labels          -> student_label (scalar)
        teacher_label   -> teacher target (scalar)
    """
    def __init__(self, data_path, tokenizer, rule_path, max_length: int = 512):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_length = max_length  # 保存max_length参数
        # Teacher = normalized RuleReward
        self.teacher = RuleReward(rule_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    # 取 assistant 回复（兼容messages/response两种格式）
                    assistant_text = None
                    # 处理messages格式
                    if "messages" in item:
                        for msg in item.get("messages", []):
                            if msg.get("role") == "assistant":
                                assistant_text = msg.get("content", "").strip()
                                break
                    # 处理response格式
                    elif "response" in item:
                        assistant_text = item.get("response", "").strip()
                    
                    if not assistant_text:
                        print(f"[WARN] line {line_id} skipped: no assistant content")
                        continue
                    # Teacher label (normalized, [-1, 0])
                    teacher_res = self.teacher.evaluate(assistant_text)
                    teacher_label = float(teacher_res["total_penalty"])
                    # Student hard label（若不存在，回退到 teacher）
                    if (
                        "rule_annotation" in item and
                        "total_penalty" in item["rule_annotation"]
                    ):
                        student_label = float(item["rule_annotation"]["total_penalty"])
                    else:
                        student_label = teacher_label
                    self.samples.append({
                        "text": assistant_text,
                        "student_label": student_label,
                        "teacher_label": teacher_label
                    })
                except Exception as e:
                    print(f"[WARN] line {line_id} skipped: {e}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length 
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            # ✔ HF Trainer 期望的 labels（标量）
            "labels": torch.tensor(s["student_label"], dtype=torch.float32),
            # ✔ 额外字段，需要自定义collator保留
            "teacher_label": torch.tensor(s["teacher_label"], dtype=torch.float32),
        }
# ===========================================================
# 2. 蒸馏 Trainer（兼容 HF Trainer 机制）
# ===========================================================
class DistillationTrainer(Trainer):
    """
    Loss = α * HardLoss + (1 - α) * DistillLoss
    α 随训练步数线性 warmup
    """
    def __init__(self, *args, warmup_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
        
    def pairwise_ranking_loss(self, preds, teacher_scores, margin=0.0, eps=0.05):
        loss = 0.0
        cnt = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                delta = teacher_scores[i] - teacher_scores[j]
                if torch.abs(delta) < eps:
                    continue
                if delta < 0:
                    loss += torch.relu(margin + preds[i] - preds[j])
                else:
                    loss += torch.relu(margin + preds[j] - preds[i])
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=preds.device)
        return loss / cnt
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ✔ 正确拆分
        student_target = inputs.pop("labels")          # [B]
        teacher_target = inputs.pop("teacher_label")   # [B]
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1)              # [B]
        raw_preds = outputs.logits.squeeze(-1)
        preds = -torch.sigmoid(raw_preds)
        # ---------------------------------------------------
        # (1) Hard loss：拟合人工 / 规则标注
        # ---------------------------------------------------
        hard_loss = torch.nn.functional.smooth_l1_loss(
            preds, student_target
        )
        # ---------------------------------------------------
        # (2) Distill loss：对齐规则 teacher（加权）
        # ---------------------------------------------------
        weight = torch.abs(teacher_target) + 0.1
        distill_loss = (weight * (preds - teacher_target) ** 2).mean()
        
        # (3) Pairwise ranking distillation loss（核心）
        rank_loss = self.pairwise_ranking_loss(
            preds,
            teacher_target
        )
        # ---------------------------------------------------
        # (4) 动态权重调度（warmup）
        # ---------------------------------------------------
        step = max(1, self.state.global_step)
        alpha = min(1.0, step / self.warmup_steps)
        beta = 0.3  #（排序权重）
        rank_weight=beta*alpha
        total_loss = (
            alpha * hard_loss +
            (1.0 - alpha) * distill_loss +
            rank_weight * rank_loss
        )
        return (total_loss, outputs) if return_outputs else total_loss
# ===========================================================
# 3. Metrics（只评估 student 拟合）
# ===========================================================
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
        "spearman": float(spear)
    }
# ===========================================================
# 4. 主训练流程
# ===========================================================
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
    args = parser.parse_args()
    print(f"Transformers version: {transformers.__version__}")
    print(f"Using base model: {args.base_model}")
    print(f"Training data: {args.train_data}")
    print(f"Max sequence length: {args.max_length}")  # 打印max_length参数，方便验证
    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        problem_type="regression"
    )
    # 设置pad token（避免tokenizer报错）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    # 加载数据集
    dataset = DistilledRuleRewardDataset(
        args.train_data, 
        tokenizer, 
        args.rule_path,
        max_length=args.max_length  # 【修改4】传递max_length参数到Dataset
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Please check your training data path and format.")
    train_ds = DistilledRuleRewardDataset(
        args.train_data,
        tokenizer,
        args.rule_path,
        max_length=args.max_length  # 【修改5】训练集传递max_length
    )
    val_ds = DistilledRuleRewardDataset(
        args.val_data,
        tokenizer,
        args.rule_path,
        max_length=args.max_length  # 【修改6】验证集传递max_length
    )
    print("Dataset loaded:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    # 自定义DataCollator（保留teacher_label）
    data_collator = CustomDataCollatorWithPadding(
        tokenizer=tokenizer
    )
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",        # ✔ 4.57.1 正确参数
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
        # 新增：避免梯度累积警告
        gradient_accumulation_steps=1,
        # 新增：训练进度条
        disable_tqdm=False,
        remove_unused_columns=False
    )
    # 初始化蒸馏训练器
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,  # 使用自定义collator
        compute_metrics=compute_metrics,
        warmup_steps=args.warmup_steps
    )
    print("===== Start Rule Reward Distillation Training =====")
    trainer.train()
    # 保存最终模型
    final_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nTraining completed!")
    print(f"Model saved to {final_dir}")
    print("NOTE: Test set is reserved for post-hoc rule fidelity evaluation.")
if __name__ == "__main__":
    main()