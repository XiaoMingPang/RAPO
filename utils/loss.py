from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Dict, Any, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================================================
# 全局控制：utility loss 低频（原 UTILITY_INTERVAL=64 每步更新）
# =========================================================
UTILITY_INTERVAL = 1
_GLOBAL_UTILITY_STEP = 0

def should_compute_utility():
    global _GLOBAL_UTILITY_STEP
    _GLOBAL_UTILITY_STEP += 1
    return (_GLOBAL_UTILITY_STEP % UTILITY_INTERVAL) == 0

# ======================== 规则惩罚模型 ========================
class RulePenaltyModel:
    def __init__(self, model_path):
        self.rule_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.rule_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.rule_model.eval()
        self.device = torch.device("cpu")
        self.rule_model.to(self.device)
        
        if self.rule_tokenizer.pad_token is None:
            self.rule_tokenizer.pad_token = self.rule_tokenizer.eos_token
            self.rule_model.config.pad_token_id = self.rule_tokenizer.pad_token_id
        self.rule_tokenizer.padding_side = "right"
        self.rule_tokenizer.truncation_side = "right"
    
    @torch.no_grad()
    def get_penalty_batch(self, texts: List[str]) -> List[float]:
        inputs = self.rule_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        penalties = self.rule_model(**inputs).logits.squeeze(-1)
        rewards = (penalties).cpu().tolist()
        return rewards

# 工具函数
def get_batch_loss(output, labels):
    return -get_log_probs(output, labels)

def get_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    """
    return: [B] mean log-prob per valid token
    """
    token_log_probs = get_log_probs(logits, labels)

    valid_mask = (labels[..., 1:] != -100).float()
    token_count = valid_mask.sum(dim=-1).clamp_min(1.0)

    seq_log_probs = token_log_probs.sum(dim=-1) / token_count
    return seq_log_probs


def get_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, T, V]
    labels: [B, T]
    return: [B, T-1] token-level log_probs
    """
    labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()

    valid_mask = labels != -100
    labels_clamped = labels.clamp(min=0)

    target_logits = torch.gather(
        logits,
        dim=-1,
        index=labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    logsumexp = torch.logsumexp(logits, dim=-1)

    log_probs = target_logits - logsumexp
    log_probs = log_probs * valid_mask

    return log_probs


def dpo_loss(main_model, ref_model, inputs, beta=0.5, retain_loss=True, alpha=0.2):
    device = next(main_model.parameters()).device
    preferred_input_ids = inputs['dpo_preferred_input_ids'].to(device)
    preferred_labels = inputs['dpo_preferred_labels'].to(device)
    preferred_attention_mask = inputs['dpo_preferred_attention_mask'].to(device)
    non_preferred_input_ids = inputs['dpo_non_preferred_input_ids'].to(device)
    non_preferred_labels = inputs['dpo_non_preferred_labels'].to(device)
    non_preferred_attention_mask = inputs['dpo_non_preferred_attention_mask'].to(device)

    preferred_outputs_main = main_model(preferred_input_ids, attention_mask=preferred_attention_mask)
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        preferred_outputs_ref = ref_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    preferred_log_probs_main = get_sequence_log_probs(preferred_outputs_main.logits, preferred_labels)
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    preferred_log_probs_ref = get_sequence_log_probs(preferred_outputs_ref.logits, preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    loss_val = -F.logsigmoid(beta * ((preferred_log_probs_main - preferred_log_probs_ref) -
                                     (non_preferred_log_probs_main - non_preferred_log_probs_ref))) * 2 / beta
    loss_val = loss_val.mean()

    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val

def npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2, retain_loss=True):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)

    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    loss_val = loss_val.mean()

    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device)
        utility_labels = inputs['utility_labels'].to(device)
        utility_attention_mask = inputs['utility_attention_mask'].to(device)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        loss_val = alpha * loss_val + (1 - alpha) * utility_loss

    return loss_val

def gd_loss(model, inputs, alpha=0.2):
    safety_input_ids = inputs['input_ids'].to(model.device)
    safety_labels = inputs['labels'].to(model.device)
    safety_attention_mask = inputs['attention_mask'].to(model.device)

    utility_input_ids = inputs['utility_input_ids'].to(model.device)
    utility_labels = inputs['utility_labels'].to(model.device)
    utility_attention_mask = inputs['utility_attention_mask'].to(model.device)

    safety_outputs = model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = get_batch_loss(safety_logits, safety_labels)

    utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_logits = utility_outputs.logits
    utility_loss = get_batch_loss(utility_logits, utility_labels)

    total_loss = alpha * torch.mean(safety_loss) + (1 - alpha) * torch.mean(utility_loss)
    return total_loss

def ga_loss(model, inputs, alpha=0.2):
    safety_input_ids = inputs['input_ids'].to(model.device)
    safety_labels = inputs['labels'].to(model.device)
    safety_attention_mask = inputs['attention_mask'].to(model.device)
    utility_input_ids = inputs['utility_input_ids'].to(model.device)
    utility_labels = inputs['utility_labels'].to(model.device)
    utility_attention_mask = inputs['utility_attention_mask'].to(model.device)
    safety_outputs = model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = -get_batch_loss(safety_logits, safety_labels)
    utility_outputs = model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_logits = utility_outputs.logits
    utility_loss = get_batch_loss(utility_logits, utility_labels)
    total_loss = alpha * torch.mean(safety_loss) + (1 - alpha) * torch.mean(utility_loss)
    return total_loss

def gd_npo_loss(main_model, ref_model, inputs, beta=0.5, alpha=0.2):
    device = next(main_model.parameters()).device
    non_preferred_input_ids = inputs["npo_input_ids"].to(device)
    non_preferred_labels = inputs["npo_labels"].to(device)
    non_preferred_attention_mask = inputs["npo_attention_mask"].to(device)
    
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    with torch.no_grad():
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
    
    non_preferred_log_probs_main = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    non_preferred_log_probs_ref = get_sequence_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)
    
    npo_loss_val = -F.logsigmoid(-beta * (non_preferred_log_probs_main - non_preferred_log_probs_ref)) * 2 / beta
    npo_loss_val = npo_loss_val.mean()
    
    safety_input_ids = inputs['input_ids'].to(device)
    safety_labels = inputs['labels'].to(device)
    safety_attention_mask = inputs['attention_mask'].to(device)
    safety_outputs = main_model(safety_input_ids, attention_mask=safety_attention_mask)
    safety_logits = safety_outputs.logits
    safety_loss = -get_sequence_log_probs(safety_logits, safety_labels)
    safety_loss = safety_loss.mean()
    
    utility_input_ids = inputs['utility_input_ids'].to(device)
    utility_labels = inputs['utility_labels'].to(device)
    utility_attention_mask = inputs['utility_attention_mask'].to(device)
    utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
    utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
    utility_loss = -utility_log_probs.mean()
    
    total_loss = alpha * (safety_loss + npo_loss_val) + (1 - alpha) * utility_loss
    return total_loss

def sigmoid_weights(dpo_model_log_probs, ref_log_probs, gamma=1.0):
    rewards = dpo_model_log_probs - ref_log_probs
    weights = 1 - torch.sigmoid(gamma * rewards)
    return weights

def exp_weights(dpo_model_log_probs, ref_log_probs, tau):
    rewards = ref_log_probs - dpo_model_log_probs
    weights = torch.exp(rewards / tau)
    return weights

def wgdnpo_loss(main_model, ref_model, dpo_model, inputs, beta=0.5, retain_loss=True, alpha=0.2,
                adaptive_method='sigmoid', gamma=1.0, tau=5.0):
    device = next(main_model.parameters).device
    torch.set_grad_enabled(True)
    dtype = next(main_model.parameters()).dtype
    
    preferred_input_ids = inputs['dpo_preferred_input_ids'].to(device, non_blocking=True)
    preferred_labels = inputs['dpo_preferred_labels'].to(device, non_blocking=True)
    preferred_attention_mask = inputs['dpo_preferred_attention_mask'].to(device, non_blocking=True)
    non_preferred_input_ids = inputs['dpo_non_preferred_input_ids'].to(device, non_blocking=True)
    non_preferred_labels = inputs['dpo_non_preferred_labels'].to(device, non_blocking=True)
    non_preferred_attention_mask = inputs['dpo_non_preferred_attention_mask'].to(device, non_blocking=True)

    preferred_outputs_main = main_model(preferred_input_ids, attention_mask=preferred_attention_mask)
    non_preferred_outputs_main = main_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)

    with torch.no_grad():
        preferred_outputs_ref = ref_model(preferred_input_ids, attention_mask=preferred_attention_mask)
        non_preferred_outputs_ref = ref_model(non_preferred_input_ids, attention_mask=non_preferred_attention_mask)
        
        dpo_log_probs = None
        adaptive_weights = None
        if adaptive_method in ['sigmoid', 'exp']:
            dpo_outputs = dpo_model(preferred_input_ids, attention_mask=preferred_attention_mask)
            dpo_log_probs = get_log_probs(dpo_outputs.logits, preferred_labels)
            preferred_log_probs_ref = get_log_probs(preferred_outputs_ref.logits, preferred_labels)
            if adaptive_method == 'sigmoid':
                adaptive_weights = sigmoid_weights(dpo_log_probs, preferred_log_probs_ref, gamma)
            elif adaptive_method == 'exp':
                adaptive_weights = exp_weights(dpo_log_probs, preferred_log_probs_ref, tau)
            
            adaptive_weights = adaptive_weights.detach()

    preferred_log_probs_main = get_log_probs(preferred_outputs_main.logits, preferred_labels)
    non_preferred_log_probs_main = get_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    if adaptive_weights is None:
        preferred_log_probs_ref = get_log_probs(preferred_outputs_ref.logits, preferred_labels)
    non_preferred_log_probs_ref = get_log_probs(non_preferred_outputs_ref.logits, non_preferred_labels)

    weighted_log_probs = adaptive_weights * preferred_log_probs_main
    gd_loss_val = -weighted_log_probs.sum(dim=-1).mean()

    seq_lengths = torch.sum(non_preferred_attention_mask, dim=1)
    non_preferred_log_probs_main_seq = get_sequence_log_probs(non_preferred_outputs_main.logits, non_preferred_labels)
    normalized_log_probs = non_preferred_log_probs_main_seq / seq_lengths
    npo_loss_val = -F.logsigmoid(-beta * normalized_log_probs - gamma)
    npo_loss_val = npo_loss_val.sum(dim=-1).mean()

    wgdnpo_loss_val = gd_loss_val + (npo_loss_val * 2 / beta)

    final_loss = None
    if retain_loss:
        utility_input_ids = inputs['utility_input_ids'].to(device, non_blocking=True)
        utility_labels = inputs['utility_labels'].to(device, non_blocking=True)
        utility_attention_mask = inputs['utility_attention_mask'].to(device, non_blocking=True)
        utility_outputs = main_model(utility_input_ids, attention_mask=utility_attention_mask)
        utility_log_probs = get_sequence_log_probs(utility_outputs.logits, utility_labels)
        utility_loss = -utility_log_probs.mean()
        final_loss = alpha * wgdnpo_loss_val + (1 - alpha) * utility_loss
    else:
        final_loss = wgdnpo_loss_val

    del preferred_outputs_ref, non_preferred_outputs_ref, dpo_outputs, dpo_log_probs
    del weighted_log_probs, normalized_log_probs, seq_lengths

    return final_loss

# ======================== RRSA-DPO Loss (Type15) ========================
def penalty_based_dpo_loss(
    main_model,
    ref_model,
    inputs,
    rule_model: RulePenaltyModel,
    tokenizer,
    beta=0.5,
    alpha=0.2,
    reward_temp=0.3,
    clip_magnitude=5.0
):
    device = main_model.device
    bs = inputs["dpo_preferred_input_ids"].size(0)

    pref_texts, non_texts = [], []
    for i in range(bs):
        pref = tokenizer.decode(inputs["dpo_preferred_input_ids"][i], skip_special_tokens=True)
        nonp = tokenizer.decode(inputs["dpo_non_preferred_input_ids"][i], skip_special_tokens=True)
        if "### Assistant:" in pref:
            pref_texts.append(pref.split("### Assistant:")[-1].strip())
        else:
            pref_texts.append(pref.strip())
        if "### Assistant:" in nonp:
            non_texts.append(nonp.split("### Assistant:")[-1].strip())
        else:
            non_texts.append(nonp.strip())

    if len(pref_texts) == 0 or len(non_texts) == 0:
        print("警告：提取的文本为空")
        dummy_tensor = torch.zeros(1, device=device, requires_grad=True)
        return dummy_tensor * 0

    with torch.no_grad():
        pref_penalties = torch.tensor(rule_model.get_penalty_batch(pref_texts), device=device)
        non_penalties = torch.tensor(rule_model.get_penalty_batch(non_texts), device=device)
        pref_rewards = torch.tanh(pref_penalties / 5.0)
        non_rewards = torch.tanh(non_penalties / 5.0)

    if torch.isnan(pref_rewards).any() or torch.isnan(non_rewards).any():
        print("警告：规则奖励包含NaN值")
        pref_rewards = torch.zeros_like(pref_rewards, device=device)
        non_rewards = torch.zeros_like(non_rewards, device=device)

    pref_ids = inputs["dpo_preferred_input_ids"].to(device)
    non_ids = inputs["dpo_non_preferred_input_ids"].to(device)
    pref_mask = inputs["dpo_preferred_attention_mask"].to(device)
    non_mask = inputs["dpo_non_preferred_attention_mask"].to(device)
    pref_labels = inputs["dpo_preferred_labels"].to(device)
    non_labels = inputs["dpo_non_preferred_labels"].to(device)

    all_ids = torch.cat([pref_ids, non_ids], dim=0)
    all_mask = torch.cat([pref_mask, non_mask], dim=0)

    outputs = main_model(all_ids, attention_mask=all_mask)
    logits = outputs.logits

    pref_logits = logits[:bs]
    non_logits = logits[bs:]

    pref_logits = torch.clamp(pref_logits, -1e4, 1e4)
    non_logits = torch.clamp(non_logits, -1e4, 1e4)

    pref_lp = get_sequence_log_probs(pref_logits, pref_labels)
    non_lp = get_sequence_log_probs(non_logits, non_labels)

    if torch.isnan(pref_lp).any() or torch.isnan(non_lp).any():
        print("警告：对数概率包含NaN值，使用备用计算")
        return (pref_lp * 0).sum()

    with torch.no_grad():
        ref_ids = torch.cat([pref_ids, non_ids], dim=0).cpu()
        ref_mask = torch.cat([pref_mask, non_mask], dim=0).cpu()

        ref_out = ref_model(ref_ids, attention_mask=ref_mask).logits
        ref_pref = ref_out[:bs].to(device)
        ref_non = ref_out[bs:].to(device)

        pref_lp_ref = get_sequence_log_probs(ref_pref, pref_labels)
        non_lp_ref = get_sequence_log_probs(ref_non, non_labels)

    reward_diff = pref_rewards - non_rewards
    reward_std = reward_diff.std()
    if reward_std < 1e-8:
        reward_diff = reward_diff / 1e-8
    else:
        reward_diff = reward_diff / (reward_std + 1e-8)
    
    if bs == 1:
        weights = torch.ones_like(reward_diff)
    else:
        weights = torch.softmax(
            torch.clamp(reward_diff / (reward_temp + 1e-8), -10, 10),
            dim=0
        )

    logit_diff = (pref_lp - pref_lp_ref) - (non_lp - non_lp_ref)
    logit_diff = torch.clamp(logit_diff, -clip_magnitude, clip_magnitude)

    if torch.isnan(logit_diff).any():
        print("警告：logit_diff包含NaN值")
        logit_diff = torch.zeros_like(logit_diff, device=device)

    dpo_loss_val = -F.logsigmoid(beta * logit_diff)
    dpo_loss_val = (dpo_loss_val * weights).mean()

    if torch.isnan(dpo_loss_val):
        print("警告：DPO loss为NaN，使用备用值")
        dpo_loss_val = (pref_lp * 0).sum()

    utility_loss = (pref_lp * 0).sum()
    
    if "utility_input_ids" in inputs:
        utility_outputs = main_model(
            inputs["utility_input_ids"].to(device),
            attention_mask=inputs["utility_attention_mask"].to(device)
        )
        utility_lp = get_sequence_log_probs(
            utility_outputs.logits,
            inputs["utility_labels"].to(device)
        )
        utility_loss = -utility_lp.mean()
        
        if not should_compute_utility():
            utility_loss = utility_loss.detach()

    # ✅ 修复：alpha 控制 dpo_loss 权重，与原版 DPO 含义一致
    total_loss = (1 - alpha) * utility_loss + alpha * dpo_loss_val

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("警告：总损失为NaN或inf，使用备用值")
        total_loss = (pref_lp * 0).sum()

    return total_loss

# ======================== RRSA-WDOOR Loss (Type14) ========================
def penalty_based_wdoor_loss(
    main_model,
    ref_model,
    inputs,
    rule_model: RulePenaltyModel,
    tokenizer,
    beta=1.0, 
    alpha=0.2
):
    device = main_model.device
    bs = inputs['dpo_preferred_input_ids'].size(0)

    pref_texts, non_texts = [], []
    for i in range(bs):
        pref = tokenizer.decode(
            inputs['dpo_preferred_input_ids'][i],
            skip_special_tokens=True
        ).split("### Assistant:")[-1].strip()
        nonp = tokenizer.decode(
            inputs['dpo_non_preferred_input_ids'][i],
            skip_special_tokens=True
        ).split("### Assistant:")[-1].strip()
        pref_texts.append(pref)
        non_texts.append(nonp)

    with torch.no_grad():
        pref_rewards = torch.tensor(
            rule_model.get_penalty_batch(pref_texts),
            device=device
        )
        non_rewards = torch.tensor(
            rule_model.get_penalty_batch(non_texts),
            device=device
        )

        # ✅ 修复：1 - sigmoid(rewards)
        # safe 文本 reward≈0 → 1-sigmoid(0)=0.5 → 高权重，鼓励 safe
        # unsafe 文本 reward≈-9 → 1-sigmoid(-9)≈1.0 → 极高权重，强烈惩罚
        pref_weights = 1 - torch.sigmoid(pref_rewards)
        non_weights = 1 - torch.sigmoid(non_rewards)

    pref_ids = inputs['dpo_preferred_input_ids'].to(device)
    non_ids = inputs['dpo_non_preferred_input_ids'].to(device)
    pref_mask = inputs['dpo_preferred_attention_mask'].to(device)
    non_mask = inputs['dpo_non_preferred_attention_mask'].to(device)
    pref_labels = inputs['dpo_preferred_labels'].to(device)
    non_labels = inputs['dpo_non_preferred_labels'].to(device)

    all_ids = torch.cat([pref_ids, non_ids], dim=0)
    all_mask = torch.cat([pref_mask, non_mask], dim=0)

    logits = main_model(all_ids, attention_mask=all_mask).logits
    pref_logits, non_logits = logits[:bs], logits[bs:]

    pref_lp = get_sequence_log_probs(pref_logits, pref_labels)
    non_lp = get_sequence_log_probs(non_logits, non_labels)

    # ✅ 修复：GD 和 NPO 双目标均使用规则加权
    # GD：鼓励 safe response
    gd_loss = (pref_weights * pref_lp).mean()
    # NPO：惩罚 unsafe response（原本用 logit 直接算，现在用规则加权）
    npo_loss = -(non_weights * non_lp).mean()

    wdoor_loss = gd_loss + npo_loss

    utility_loss = (pref_lp * 0).sum()
    if should_compute_utility() and 'utility_input_ids' in inputs:
        utility_outputs = main_model(
            inputs['utility_input_ids'].to(device),
            attention_mask=inputs['utility_attention_mask'].to(device)
        )
        utility_lp = get_sequence_log_probs(
            utility_outputs.logits,
            inputs['utility_labels'].to(device)
        )
        utility_loss = -utility_lp.mean()

    # ✅ 修复：alpha 含义与原版 WDOOR 一致
    total_loss = (1 - alpha) * utility_loss + alpha * wdoor_loss
    return total_loss


# ======================== Trainer 类 ========================
class GATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return ga_loss(model, inputs)

class NPOTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model 
    def compute_loss(self, model, inputs, return_outputs=False):
        return npo_loss(model, self.ref_model, inputs)

class GDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return gd_loss(model, inputs)

class DPOTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return dpo_loss(model, self.ref_model, inputs)

class DOORTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model 
    def compute_loss(self, model, inputs, return_outputs=False):
        return gd_npo_loss(model, self.ref_model, inputs)

class WDOORSIGTrainer(Trainer):
    def __init__(self, ref_model, dpo_model, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.dpo_model = dpo_model
        self.gamma = gamma
    def compute_loss(self, model, inputs, return_outputs=False):
        return wgdnpo_loss(model, self.ref_model, self.dpo_model, inputs, gamma=self.gamma, adaptive_method='sigmoid')

class WDOORTrainer(Trainer):
    def __init__(self, ref_model, dpo_model, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.dpo_model = dpo_model
        self.tau = tau
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return wgdnpo_loss(model, self.ref_model, self.dpo_model, inputs, adaptive_method='exp', tau=self.tau)


class PenaltyDPOTrainer(Trainer):
    def __init__(self, ref_model, rule_model: RulePenaltyModel, tokenizer, 
                 beta=0.5, alpha=0.2, reward_temp=0.3, clip_magnitude=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.rule_model = rule_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.alpha = alpha
        self.reward_temp = reward_temp
        self.clip_magnitude = clip_magnitude

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = penalty_based_dpo_loss(
            model, self.ref_model, inputs, 
            self.rule_model, self.tokenizer,
            beta=self.beta, 
            alpha=self.alpha,
            reward_temp=self.reward_temp,
            clip_magnitude=self.clip_magnitude
        )
        return (loss, None) if return_outputs else loss


class PenaltyWDOORTrainer(Trainer):
    def __init__(
        self,
        ref_model,
        rule_model: RulePenaltyModel,
        tokenizer,
        beta=1.0,
        alpha=0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.rule_model = rule_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = penalty_based_wdoor_loss(
            model,
            self.ref_model,
            inputs,
            self.rule_model,
            self.tokenizer,
            beta=self.beta,
            alpha=self.alpha
        )
        return (loss, None) if return_outputs else loss