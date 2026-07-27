from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.custom_dataset import CustomDataset

from utils.loss import (
    GATrainer, NPOTrainer, GDTrainer, DPOTrainer, 
    DOORTrainer, WDOORTrainer, 
    PenaltyDPOTrainer, PenaltyDOORTrainer, PenaltyWDOORTrainer,
    PenaltyDistillationModel,
)  


import torch
import argparse
import warnings
import json  

from transformers import BitsAndBytesConfig
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--base_model', type=str, default="./gemma-2-2b-it")
parser.add_argument('--dataset_model_name', type=str, default="gemma-2-2b-it")  
parser.add_argument('--output_dir', type=str, default="output_model")
parser.add_argument(
    "--penalty_strength",
    type=float,
    default=1.0,
)
# 训练类型：9=原DPO | 12=原WDOOR | 14=RAPO-DPO | 15=RAPO-DOOR | 16=RAPO-WDOOR
parser.add_argument('--type', type=int, default=14,
                    help="训练类型：9=原DPO | 12=原WDOOR | 14=RAPO-DPO | 15=RAPO-DOOR | 16=RAPO-WDOOR")
# 规则模型路径（仅 RAPO 类型需要）
parser.add_argument('--rule_model_path', type=str, default="./rule_penalty_model/final_model/")
# RAPO 专用：规则模型最大编码长度
parser.add_argument('--penalty_max_length', type=int, default=512,
                    help="PenaltyDistillationModel 的最大序列长度")
parser.add_argument('--penalty_device', type=str, default="cuda:0",
                    help="PenaltyDistillationModel 所在设备")


args = parser.parse_args()

# ======================== 1. 加载规则惩罚模型（仅 RAPO 类型） ========================
penalty_model = None
if args.type in [14, 15, 16]:
    try:
        penalty_model = PenaltyDistillationModel(
            model_path=args.rule_model_path,
            device=args.penalty_device,
            max_length=args.penalty_max_length,
        )
        print(f"✅ 成功加载规则惩罚模型: {args.rule_model_path} "
              f"(device={args.penalty_device}, max_length={args.penalty_max_length})")
    except Exception as e:
        raise RuntimeError(f"❌ 规则惩罚模型加载失败: {e}")
else:
    print("ℹ️ 非 RAPO 训练类型，跳过规则惩罚模型加载")

# ======================== 2. 加载主模型和Tokenizer ========================
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": "cuda:0"}
)

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
# Tokenizer配置（Gemma专属）
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# ======================== 3. 加载参考模型（Ref Model） ========================
ref_model = None
dpo_model = None  

if args.type in [4,5,7,9, 10,12, 14, 15, 16]:
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
#        load_in_8bit=True,
        quantization_config=bnb_4bit
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("✅ 参考模型加载完成（梯度已冻结）")

# ======================== 4. 加载WDOOR专用的DPO模型（仅原WDOOR需要） ========================
if args.type == 12:   # 仅原 WDOOR(12) 需要，RAPO 系列不需要
    try:
        dpo_name = f"./output_model/dpo_model/{args.dataset_model_name}/9"  # 动态适配模型名
        dpo_model = AutoModelForCausalLM.from_pretrained(
            dpo_name,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
#            load_in_8bit=True
            quantization_config=bnb_4bit,
        )
        dpo_model.eval()
        for param in dpo_model.parameters():
            param.requires_grad = False
        print(f"✅ WDOOR专用DPO模型加载完成: {dpo_name}")
    except Exception as e:
        raise RuntimeError(f"❌ WDOOR专用DPO模型加载失败: {e}")
elif args.type in [14, 15, 16]:
    print("ℹ️ RAPO 系列无需加载预训练 DPO 模型")

# ======================== 5. 数据集配置（四类模型共用统一格式） ========================
train_dataset_dict = {}
# RAPO 系列（14/15/16）和 原 DPO/WDOOR（9/12）使用同一份 preference + utility 数据
if args.type in [9, 12, 14, 15, 16]:
    train_dataset_dict = {
        "dpo_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
        "dpo_non_preferred_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
        "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
    }
# 保留基础类型（1-8/10）兼容
else:
    if args.type == 1:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 2:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 3:
        train_dataset_dict = {
            "dpo_preferred_": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "dpo_non_preferred_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 4:
        train_dataset_dict = {
            "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 5:
        train_dataset_dict = {
            "": (f"./dataset/train/{args.dataset_model_name}-good.jsonl", False),
            "npo_": (f"./dataset/train/{args.dataset_model_name}-bad.jsonl", False),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 6:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 7:
        train_dataset_dict = {
            "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 8:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }
    elif args.type == 10:
        train_dataset_dict = {
            "": (f"./dataset/prefix/{args.dataset_model_name}-prefix-good.jsonl", True),
            "npo_": (f"./dataset/prefix/{args.dataset_model_name}-prefix-bad.jsonl", True),
            "utility_": ("./dataset/train/alpaca_cleaned.jsonl", False)
        }

# 加载数据集
dataset = CustomDataset(tokenizer, train_dataset_dict, max_words=512)
print(f"✅ 数据集加载完成，样本数: {len(dataset)}")

# ======================== 6. 数据Collator ========================
def data_collator(batch_list):
    result = {}
    for key in batch_list[0].keys():
        if isinstance(batch_list[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch_list])
        else:
            raise ValueError(f"Unsupported type for key {key}: {type(batch_list[0][key])}")
    return result

# ======================== 7. 训练参数配置 ========================
training_arguments = TrainingArguments(
    output_dir=f"{args.output_dir}/{args.dataset_model_name}-200/{args.type}-checkpoint",
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    learning_rate=5e-6,
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="epoch",
    remove_unused_columns=False,
    save_only_model=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    report_to="none",
    log_level="error",
    dataloader_pin_memory=False,
    max_grad_norm=1.0, 
    warmup_ratio=0.1
)

# ======================== 8. Trainer选择逻辑（核心：4类模型） ========================
trainer = None
# 基础类型（保留兼容）
if args.type == 1:
    trainer = GATrainer(model=model, train_dataset=dataset, args=training_arguments,
                        data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 2:
    trainer = NPOTrainer(model=model, train_dataset=dataset, args=training_arguments,
                         ref_model=ref_model, data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 3:
    trainer = GDTrainer(model=model, train_dataset=dataset, args=training_arguments,
                        data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 4:
    trainer = NPOTrainer(model=model, train_dataset=dataset, args=training_arguments,
                         ref_model=ref_model, data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 5:
    trainer = DOORTrainer(model=model, train_dataset=dataset, args=training_arguments,
                          ref_model=ref_model, data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 6:
    trainer = GATrainer(model=model, train_dataset=dataset, args=training_arguments,
                        data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 7:
    trainer = NPOTrainer(model=model, train_dataset=dataset, args=training_arguments,
                         ref_model=ref_model, data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 8:
    trainer = GDTrainer(model=model, train_dataset=dataset, args=training_arguments,
                        data_collator=data_collator, tokenizer=tokenizer)
elif args.type == 10:
    trainer = DOORTrainer(model=model, train_dataset=dataset, args=training_arguments,
                          ref_model=ref_model, data_collator=data_collator, tokenizer=tokenizer)

# 核心：原DPO模型（type=9）
elif args.type == 9:
    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

# 核心：原WDOOR模型（type=12）
elif args.type == 12:
    trainer = WDOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        dpo_model=dpo_model,
        data_collator=data_collator,
        tau=5.0,
        tokenizer=tokenizer
    )

# RAPO-DPO（type=14）
elif args.type == 14:
    trainer = PenaltyDPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        penalty_model=penalty_model,
        tokenizer=tokenizer,
        beta=0.5,
        alpha=0.2,
        penalty_strength=args.penalty_strength,
        data_collator=data_collator,
    )
# RAPO-DOOR（type=15）
elif args.type == 15:
    trainer = PenaltyDOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        penalty_model=penalty_model,
        tokenizer=tokenizer,
        beta=0.5,
        alpha=0.2,
        penalty_strength=args.penalty_strength,
        data_collator=data_collator
    )
    
# RAPO-WDOOR（type=16）
elif args.type == 16:
    trainer = PenaltyWDOORTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        ref_model=ref_model,
        penalty_model=penalty_model,
        tokenizer=tokenizer,
        beta=0.5,
        alpha=0.2,
        penalty_strength=args.penalty_strength,
        data_collator=data_collator
    )


else:
    raise ValueError(
        f"❌ 不支持的训练类型: {args.type} "
        f"（支持：9=原DPO | 12=原WDOOR | 14=RAPO-DPO | 15=RAPO-DOOR | 16=RAPO-WDOOR）"
    )

# ======================== 9. 开始训练 ========================
# 打印训练信息（区分四类模型）
train_type_desc = {
    9: "原 DPO 模型",
    12: "原 WDOOR 模型",
    14: "RAPO-DPO 模型",
    15: "RAPO-DOOR 模型",
    16: "RAPO-WDOOR 模型",
}.get(args.type, f"基础类型 {args.type}")

print(f"\n🚀 开始训练（类型: {args.type} | 训练目标: {train_type_desc}）")
print(f"📁 输出目录: {training_arguments.output_dir}")
trainer.train()

# ======================== 10. 保存最终模型 ========================
final_output_dir = f"{args.output_dir}/{args.dataset_model_name}/{args.type}"
trainer.save_model(final_output_dir)
print(f"\n✅ 训练完成！最终模型已保存至: {final_output_dir}")

# 保存训练参数
with open(f"{final_output_dir}/training_args.json", "w") as f:
    f.write(training_arguments.to_json_string())
print(f"✅ 训练参数已保存至: {final_output_dir}/training_args.json")

# 保存 RAPO 相关参数（仅 RAPO 类型）
if args.type in [14, 15, 16]:
    rule_args = {
        "type": args.type,
        "rule_model_path": args.rule_model_path,
        "penalty_strength": args.penalty_strength,
        "penalty_max_length": args.penalty_max_length,
        "penalty_device": args.penalty_device,
        "beta": 0.5,
        "alpha": 0.2,
    }
    with open(f"{final_output_dir}/rule_args.json", "w") as f:
        json.dump(rule_args, f, indent=4, ensure_ascii=False)
    print(f"✅ RAPO 参数已保存至: {final_output_dir}/rule_args.json")