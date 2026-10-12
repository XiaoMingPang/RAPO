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

# [RAPO-FIX1] 诊断开关：由 train.py --penalty_debug 打开
PENALTY_DEBUG = False
_PENALTY_DEBUG_STEP = 0

def _maybe_log_risk_gap(preferred_penalties, rejected_penalties):
    """每 20 步打印 risk_gap 统计，验证惩罚信号是否有效（risk_gap 应显著 >0）。"""
    global _PENALTY_DEBUG_STEP
    if not PENALTY_DEBUG:
        return
    _PENALTY_DEBUG_STEP += 1
    if _PENALTY_DEBUG_STEP % 20 != 0:
        return
    with torch.no_grad():
        preferred_risk = -preferred_penalties
        rejected_risk = -rejected_penalties
        risk_gap = (rejected_risk - preferred_risk).clamp(min=0.0, max=1.0)
        print(
            f"[PENALTY-DEBUG] step={_PENALTY_DEBUG_STEP} "
            f"pref_pen={preferred_penalties.mean().item():+.4f} "
            f"rej_pen={rejected_penalties.mean().item():+.4f} "
            f"risk_gap_mean={risk_gap.mean().item():.4f} "
            f"risk_gap_zero={(risk_gap == 0).float().mean().item() * 100:.1f}%"
        )

def should_compute_utility():
    global _GLOBAL_UTILITY_STEP
    _GLOBAL_UTILITY_STEP += 1
    return (_GLOBAL_UTILITY_STEP % UTILITY_INTERVAL) == 0

# ======================== 规则惩罚蒸馏模型 ========================
class PenaltyDistillationModel:
    def __init__(
        self,
        model_path,
        device="cuda:0",
        max_length=512,
        score_temperature=1.0,
    ):
        # [RAPO-FIX-Q2] score_temperature > 1 时 sigmoid 分数不饱和，
        # risk_gap 随样本变化，保住「动态风险边际」的多样性（默认 1.0 不改变行为）
        self.score_temperature = float(score_temperature)
        self.penalty_tokenizer = (
            AutoTokenizer.from_pretrained(model_path)
        )
        self.penalty_model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
        )

        self.device = torch.device(device)
        self.max_length = max_length

        if self.penalty_tokenizer.pad_token is None:
            if self.penalty_tokenizer.eos_token is None:
                raise ValueError(
                    "Penalty tokenizer must define a "
                    "pad token or EOS token."
                )

            self.penalty_tokenizer.pad_token = (
                self.penalty_tokenizer.eos_token
            )
            self.penalty_model.config.pad_token_id = (
                self.penalty_tokenizer.pad_token_id
            )

        self.penalty_tokenizer.padding_side = "right"
        self.penalty_tokenizer.truncation_side = "right"

        self.penalty_model.to(self.device)
        self.penalty_model.eval()
        self.penalty_model.requires_grad_(False)

    @torch.no_grad()
    def get_penalty_batch(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        if not texts:
            raise ValueError(
                "Penalty model received an empty batch."
            )

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Penalty model received an empty response."
            )

        inputs = self.penalty_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        raw_logits = (
            self.penalty_model(**inputs)
            .logits
            .squeeze(-1)
            .float()
        )

        # [RAPO-FIX1+2] 惩罚模型改为「直接分类器」（BCEWithLogits 训练，输出无界 logit），
        # 这里用 -sigmoid 映射到 (-1, 0)：
        #   安全(拒绝续写) -> logit 负 -> penalty≈0  -> risk≈0
        #   有害(有害续写) -> logit 正 -> penalty≈-1 -> risk≈1
        # 相比旧的 clamp(-1,0)（把弱回归信号压平），sigmoid 映射更稳、更可区分。
        # [RAPO-FIX-Q2] 除以 score_temperature（>1 时分数不饱和，risk_gap 更分散，
        # 使 margin 保持样本级动态性）
        penalties = -torch.sigmoid(raw_logits / self.score_temperature)

        if penalties.ndim == 0:
            penalties = penalties.unsqueeze(0)

        expected_shape = (len(texts),)

        if penalties.shape != expected_shape:
            raise ValueError(
                f"Expected penalty shape {expected_shape}, "
                f"got {tuple(penalties.shape)}."
            )

        if not torch.isfinite(penalties).all():
            raise FloatingPointError(
                "Penalty model returned non-finite values."
            )

        return penalties

# 工具函数
def get_batch_loss(output, labels):
    return -get_log_probs(output, labels)

#def get_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor):
#    """
#    return: [B] mean log-prob per valid token
#    """
#    token_log_probs = get_log_probs(logits, labels)
#
#    valid_mask = (labels[..., 1:] != -100).float()
#    token_count = valid_mask.sum(dim=-1).clamp_min(1.0)
#
#    seq_log_probs = token_log_probs.sum(dim=-1) / token_count
#    return seq_log_probs
def get_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    log_probs = -loss_function(logits.transpose(-1, -2), shifted_labels)
    sequence_log_probs = log_probs.sum(dim=-1)
    return sequence_log_probs
    

def get_log_probs(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 128):
    """
    logits: [B, T, V]
    labels: [B, T]
    return: [B, T-1] token-level log_probs
    
    沿 sequence 维分块计算 log_probs，数学上与原版完全等价。
    节省显存：避免一次性分配 [B, T-1, V] 的中间张量。
    """
    T_minus_1 = logits.size(1) - 1  # 预测的 token 数 = T - 1
    
    log_probs_chunks = []
    for start in range(0, T_minus_1, chunk_size):
        end = min(start + chunk_size, T_minus_1)
        # logits[:, i] 预测 labels[:, i+1]
        chunk_logits = logits[:, start:end, :]                     # [B, chunk, V]
        chunk_labels = labels[:, start + 1:end + 1].contiguous()   # [B, chunk]
        
        valid_mask = (chunk_labels != -100).float()
        labels_clamped = chunk_labels.clamp(min=0)
        
        target_logits = torch.gather(
            chunk_logits,
            dim=-1,
            index=labels_clamped.unsqueeze(-1),
        ).squeeze(-1)  # [B, chunk]
        
        logsumexp = torch.logsumexp(chunk_logits, dim=-1)  # [B, chunk]
        chunk_lp = (target_logits - logsumexp) * valid_mask
        log_probs_chunks.append(chunk_lp)
    
    return torch.cat(log_probs_chunks, dim=-1)
def get_rapo_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [B, T, V]."
        )

    if labels.ndim != 2:
        raise ValueError(
            "labels must have shape [B, T]."
        )

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must have matching "
            "batch and sequence dimensions."
        )

    shift_logits = (
        logits[..., :-1, :]
        .contiguous()
        .float()
    )

    shift_labels = (
        labels[..., 1:]
        .contiguous()
    )

    valid_mask = shift_labels != -100
    token_count = valid_mask.sum(dim=-1)

    # ===== 容错处理：跳过空样本（不崩溃） =====
    invalid_mask = (token_count == 0)
    if invalid_mask.any():
        invalid_indices = (
            invalid_mask.nonzero(as_tuple=False).flatten().tolist()
        )
        print(
            f"⚠️ Warning: Batch indices {invalid_indices} "
            f"have no response tokens. Using fallback."
        )
        # 替换 token_count 为 1，避免除零
        # 同时让该样本的 log_probs = 0（不影响 loss 平均）
        token_count = token_count.masked_fill(invalid_mask, 1)
        valid_mask = valid_mask.masked_fill(
            invalid_mask.unsqueeze(-1), False
        )

    safe_labels = shift_labels.masked_fill(
        ~valid_mask,
        0,
    )

    all_log_probs = F.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = all_log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs.masked_fill(
        ~valid_mask,
        0.0,
    )

    if not torch.isfinite(token_log_probs).all():
        raise FloatingPointError(
            "Non-finite token log-probabilities detected."
        )

    sequence_log_probs = (
        token_log_probs.sum(dim=-1)
    )

    return sequence_log_probs

def decode_rapo_responses(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
) -> List[str]:
    if input_ids.shape != labels.shape:
        raise ValueError(
            "input_ids and labels must have "
            "the same shape."
        )

    texts = []

    for sample_ids, sample_labels in zip(
        input_ids,
        labels,
    ):
        sample_ids = sample_ids.detach().cpu()
        sample_labels = sample_labels.detach().cpu()

        response_mask = sample_labels != -100
        response_ids = sample_ids[response_mask]

        if response_ids.numel() == 0:
            # ===== Fallback 1: 用整个非 padding 序列作为文本 =====
            non_pad_mask = sample_ids != tokenizer.pad_token_id
            non_pad_ids = sample_ids[non_pad_mask]
            text = tokenizer.decode(
                non_pad_ids.tolist(),
                skip_special_tokens=True,
            ).strip()
            if not text:
                text = "Response truncated due to length limit."
            texts.append(text)
            continue

        text = tokenizer.decode(
            response_ids.tolist(),
            skip_special_tokens=True,
        ).strip()

        if not text:
            # ===== Fallback 2: 空文本 fallback =====
            text = "Empty response after decoding."

        texts.append(text)

    return texts


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

def exp_weights(dpo_model_log_probs, ref_log_probs, tau,max_weight=20):
    rewards = ref_log_probs - dpo_model_log_probs
    weights = torch.exp(rewards / tau)
    return weights.clamp(max=max_weight) 


# ======================== RAPO-DPO Loss (Type14) ========================
def _format_penalty_texts(
    users: List[str],
    responses: List[str],
) -> List[str]:
    # [RAPO-FIX2] 惩罚模型打分时带上问题上下文。
    # 格式必须与 utils/train_prefill_classifier.py 训练数据的格式完全一致，
    # 否则分布漂移会导致 risk_gap 失真。
    return [
        f"Question: {u}\nResponse: {r}"
        for u, r in zip(users, responses)
    ]

def _get_batch_users(inputs, key, batch_size):
    # 兼容旧数据（无 user 字段时回退为空字符串）
    users = inputs.get(key, None)
    if users is None:
        users = [""] * batch_size
    return list(users)

def penalty_based_dpo_loss(
    main_model,
    ref_model,
    inputs,
    penalty_model: PenaltyDistillationModel,
    tokenizer,
    beta=0.5,
    alpha=0.2,
    penalty_strength=1.0,
    return_outputs=False,
):
    if ref_model is None:
        raise ValueError(
            "RAPO-DPO requires a reference model."
        )

    if penalty_model is None:
        raise ValueError(
            "RAPO-DPO requires a penalty model."
        )

    if beta <= 0:
        raise ValueError(
            "beta must be greater than zero."
        )

    if penalty_strength < 0:
        raise ValueError(
            "penalty_strength must be non-negative."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be in [0, 1]."
        )

    device = next(main_model.parameters()).device

    preferred_input_ids_cpu = (
        inputs["dpo_preferred_input_ids"]
    )
    rejected_input_ids_cpu = (
        inputs["dpo_non_preferred_input_ids"]
    )
    preferred_labels_cpu = (
        inputs["dpo_preferred_labels"]
    )
    rejected_labels_cpu = (
        inputs["dpo_non_preferred_labels"]
    )

    batch_size = preferred_input_ids_cpu.size(0)

    if rejected_input_ids_cpu.size(0) != batch_size:
        raise ValueError(
            "Preferred and rejected batch sizes "
            "must be equal."
        )

    preferred_texts = _format_penalty_texts(
        _get_batch_users(inputs, "dpo_preferred_user", batch_size),
        decode_rapo_responses(
            preferred_input_ids_cpu,
            preferred_labels_cpu,
            tokenizer,
        ),
    )

    rejected_texts = _format_penalty_texts(
        _get_batch_users(inputs, "dpo_non_preferred_user", batch_size),
        decode_rapo_responses(
            rejected_input_ids_cpu,
            rejected_labels_cpu,
            tokenizer,
        ),
    )

    all_penalties = (
        penalty_model.get_penalty_batch(
            preferred_texts + rejected_texts
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    expected_penalty_count = 2 * batch_size

    if all_penalties.shape != (
        expected_penalty_count,
    ):
        raise ValueError(
            "Penalty batch size does not match "
            "the preference-pair batch size."
        )

    preferred_penalties = (
        all_penalties[:batch_size]
    )
    rejected_penalties = (
        all_penalties[batch_size:]
    )

    _maybe_log_risk_gap(preferred_penalties, rejected_penalties)

    preferred_risk = -preferred_penalties
    rejected_risk = -rejected_penalties

    risk_gap = (
        rejected_risk - preferred_risk
    ).clamp(
        min=0.0,
        max=1.0,
    )

    preferred_input_ids = (
        preferred_input_ids_cpu.to(device)
    )
    rejected_input_ids = (
        rejected_input_ids_cpu.to(device)
    )
    preferred_labels = (
        preferred_labels_cpu.to(device)
    )
    rejected_labels = (
        rejected_labels_cpu.to(device)
    )
    preferred_attention_mask = (
        inputs["dpo_preferred_attention_mask"]
        .to(device)
    )
    rejected_attention_mask = (
        inputs["dpo_non_preferred_attention_mask"]
        .to(device)
    )

    pair_input_ids = torch.cat(
        [
            preferred_input_ids,
            rejected_input_ids,
        ],
        dim=0,
    )

    pair_attention_mask = torch.cat(
        [
            preferred_attention_mask,
            rejected_attention_mask,
        ],
        dim=0,
    )

    policy_outputs = main_model(
        pair_input_ids,
        attention_mask=pair_attention_mask,
    )

    preferred_policy_log_probs = (
        get_rapo_sequence_log_probs(
            policy_outputs.logits[:batch_size],
            preferred_labels,
        )
    )

    rejected_policy_log_probs = (
        get_rapo_sequence_log_probs(
            policy_outputs.logits[batch_size:],
            rejected_labels,
        )
    )

    ref_device = next(
        ref_model.parameters()
    ).device

    ref_input_ids = pair_input_ids.to(
        ref_device
    )
    ref_attention_mask = (
        pair_attention_mask.to(ref_device)
    )
    ref_preferred_labels = (
        preferred_labels.to(ref_device)
    )
    ref_rejected_labels = (
        rejected_labels.to(ref_device)
    )

    with torch.no_grad():
        ref_outputs = ref_model(
            ref_input_ids,
            attention_mask=ref_attention_mask,
        )

        preferred_ref_log_probs = (
            get_rapo_sequence_log_probs(
                ref_outputs.logits[:batch_size],
                ref_preferred_labels,
            )
            .to(device)
        )

        rejected_ref_log_probs = (
            get_rapo_sequence_log_probs(
                ref_outputs.logits[batch_size:],
                ref_rejected_labels,
            )
            .to(device)
        )

    preferred_log_ratio = (
        preferred_policy_log_probs
        - preferred_ref_log_probs
    )

    rejected_log_ratio = (
        rejected_policy_log_probs
        - rejected_ref_log_probs
    )

    dpo_log_ratio = (
        preferred_log_ratio
        - rejected_log_ratio
    )

    adaptive_safety_margin = (
        penalty_strength * risk_gap
    )

    rapo_preference_logit = (
        beta * dpo_log_ratio
        - adaptive_safety_margin
    )

    rapo_dpo_loss_per_pair = (
        (2.0 / beta)
        * F.softplus(
            -rapo_preference_logit
        )
    )

    rapo_dpo_loss = (
        rapo_dpo_loss_per_pair.mean()
    )

    if not torch.isfinite(rapo_dpo_loss):
        raise FloatingPointError(
            "Non-finite RAPO-DPO loss detected."
        )

    utility_loss = None

    if "utility_input_ids" in inputs:
        required_utility_keys = {
            "utility_input_ids",
            "utility_attention_mask",
            "utility_labels",
        }

        missing_keys = (
            required_utility_keys
            - set(inputs.keys())
        )

        if missing_keys:
            raise KeyError(
                "Missing utility inputs: "
                f"{sorted(missing_keys)}"
            )

        utility_outputs = main_model(
            inputs["utility_input_ids"].to(
                device
            ),
            attention_mask=inputs[
                "utility_attention_mask"
            ].to(device),
        )

        utility_log_probs = (
            get_rapo_sequence_log_probs(
                utility_outputs.logits,
                inputs["utility_labels"].to(
                    device
                ),
            )
        )

        utility_loss = (
            -utility_log_probs.mean()
        )

        total_loss = (
            alpha * rapo_dpo_loss
            + (1.0 - alpha) * utility_loss
        )
    else:
        total_loss = rapo_dpo_loss

    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            "Non-finite total RAPO-DPO loss detected."
        )

    if return_outputs:
        return total_loss, policy_outputs

    return total_loss
# ======================== RAPO-DOOR Loss (Type15) ========================
def penalty_based_door_loss(
    main_model,
    ref_model,
    inputs,
    penalty_model: PenaltyDistillationModel,
    tokenizer,
    beta=0.5,
    alpha=0.2,
    penalty_strength=1.0,
    return_outputs=False,
):
    if ref_model is None:
        raise ValueError(
            "RAPO-DOOR requires a reference model."
        )

    if penalty_model is None:
        raise ValueError(
            "RAPO-DOOR requires a penalty model."
        )

    if beta <= 0:
        raise ValueError(
            "beta must be greater than zero."
        )

    if penalty_strength < 0:
        raise ValueError(
            "penalty_strength must be non-negative."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be in [0, 1]."
        )

    required_pair_keys = {
        "dpo_preferred_input_ids",
        "dpo_preferred_labels",
        "dpo_preferred_attention_mask",
        "dpo_non_preferred_input_ids",
        "dpo_non_preferred_labels",
        "dpo_non_preferred_attention_mask",
    }

    missing_pair_keys = (
        required_pair_keys - set(inputs.keys())
    )

    if missing_pair_keys:
        raise KeyError(
            "Missing preference-pair inputs: "
            f"{sorted(missing_pair_keys)}"
        )

    device = next(main_model.parameters()).device

    preferred_input_ids_cpu = (
        inputs["dpo_preferred_input_ids"]
    )

    rejected_input_ids_cpu = (
        inputs["dpo_non_preferred_input_ids"]
    )

    preferred_labels_cpu = (
        inputs["dpo_preferred_labels"]
    )

    rejected_labels_cpu = (
        inputs["dpo_non_preferred_labels"]
    )

    batch_size = preferred_input_ids_cpu.size(0)

    if batch_size == 0:
        raise ValueError(
            "RAPO-DOOR received an empty batch."
        )

    if rejected_input_ids_cpu.size(0) != batch_size:
        raise ValueError(
            "Preferred and rejected batch sizes "
            "must be equal."
        )

    preferred_texts = _format_penalty_texts(
        _get_batch_users(inputs, "dpo_preferred_user", batch_size),
        decode_rapo_responses(
            preferred_input_ids_cpu,
            preferred_labels_cpu,
            tokenizer,
        ),
    )

    rejected_texts = _format_penalty_texts(
        _get_batch_users(inputs, "dpo_non_preferred_user", batch_size),
        decode_rapo_responses(
            rejected_input_ids_cpu,
            rejected_labels_cpu,
            tokenizer,
        ),
    )

    all_penalties = (
        penalty_model.get_penalty_batch(
            preferred_texts + rejected_texts
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    expected_penalty_shape = (
        2 * batch_size,
    )

    if all_penalties.shape != expected_penalty_shape:
        raise ValueError(
            f"Expected penalty shape "
            f"{expected_penalty_shape}, "
            f"got {tuple(all_penalties.shape)}."
        )

    preferred_penalties = (
        all_penalties[:batch_size]
        .clamp(
            min=-1.0,
            max=0.0,
        )
    )

    rejected_penalties = (
        all_penalties[batch_size:]
        .clamp(
            min=-1.0,
            max=0.0,
        )
    )

    _maybe_log_risk_gap(preferred_penalties, rejected_penalties)

    preferred_risk = (
        -preferred_penalties
    ).clamp(
        min=0.0,
        max=1.0,
    )

    rejected_risk = (
        -rejected_penalties
    ).clamp(
        min=0.0,
        max=1.0,
    )

    risk_gap = (
        rejected_risk
        - preferred_risk
    ).clamp(
        min=0.0,
        max=1.0,
    )

    adaptive_safety_margin = (
        penalty_strength
        * risk_gap
    )

    preferred_input_ids = (
        preferred_input_ids_cpu.to(device)
    )

    rejected_input_ids = (
        rejected_input_ids_cpu.to(device)
    )

    preferred_labels = (
        preferred_labels_cpu.to(device)
    )

    rejected_labels = (
        rejected_labels_cpu.to(device)
    )

    preferred_attention_mask = (
        inputs["dpo_preferred_attention_mask"]
        .to(device)
    )

    rejected_attention_mask = (
        inputs["dpo_non_preferred_attention_mask"]
        .to(device)
    )

    pair_input_ids = torch.cat(
        [
            preferred_input_ids,
            rejected_input_ids,
        ],
        dim=0,
    )

    pair_attention_mask = torch.cat(
        [
            preferred_attention_mask,
            rejected_attention_mask,
        ],
        dim=0,
    )

    policy_outputs = main_model(
        pair_input_ids,
        attention_mask=pair_attention_mask,
    )

    preferred_policy_log_probs = (
        get_rapo_sequence_log_probs(
            policy_outputs.logits[:batch_size],
            preferred_labels,
        )
    )

    rejected_policy_log_probs = (
        get_rapo_sequence_log_probs(
            policy_outputs.logits[batch_size:],
            rejected_labels,
        )
    )

    ref_device = next(
        ref_model.parameters()
    ).device

    with torch.no_grad():
        rejected_ref_outputs = ref_model(
            rejected_input_ids.to(ref_device),
            attention_mask=(
                rejected_attention_mask.to(
                    ref_device
                )
            ),
        )

        rejected_ref_log_probs = (
            get_rapo_sequence_log_probs(
                rejected_ref_outputs.logits,
                rejected_labels.to(ref_device),
            )
            .to(device)
        )

    rejected_log_ratio = (
        rejected_policy_log_probs
        - rejected_ref_log_probs
    )

    gd_loss_per_sample = (
        -preferred_policy_log_probs
    )

    npo_logit = (
        beta * rejected_log_ratio
        + adaptive_safety_margin
    )

    npo_loss_per_sample = (
        (2.0 / beta)
        * F.softplus(npo_logit)
    )

    gd_loss = (
        gd_loss_per_sample.mean()
    )

    npo_loss = (
        npo_loss_per_sample.mean()
    )

    rapo_door_loss = (
        gd_loss + npo_loss
    )

    if not torch.isfinite(gd_loss):
        raise FloatingPointError(
            "Non-finite RAPO-DOOR GD loss detected."
        )

    if not torch.isfinite(npo_loss):
        raise FloatingPointError(
            "Non-finite RAPO-DOOR NPO loss detected."
        )

    utility_loss = None

    utility_keys = {
        "utility_input_ids",
        "utility_attention_mask",
        "utility_labels",
    }

    present_utility_keys = (
        utility_keys & set(inputs.keys())
    )

    if present_utility_keys:
        missing_utility_keys = (
            utility_keys - set(inputs.keys())
        )

        if missing_utility_keys:
            raise KeyError(
                "Incomplete utility inputs; missing: "
                f"{sorted(missing_utility_keys)}"
            )

        utility_outputs = main_model(
            inputs["utility_input_ids"].to(
                device
            ),
            attention_mask=inputs[
                "utility_attention_mask"
            ].to(device),
        )

        utility_log_probs = (
            get_rapo_sequence_log_probs(
                utility_outputs.logits,
                inputs["utility_labels"].to(
                    device
                ),
            )
        )

        utility_loss = (
            -utility_log_probs.mean()
        )

        total_loss = (
            alpha * rapo_door_loss
            + (1.0 - alpha) * utility_loss
        )
    else:
        total_loss = rapo_door_loss

    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            "Non-finite total RAPO-DOOR loss detected."
        )

    if return_outputs:
        return total_loss, policy_outputs

    return total_loss


# ======================== Trainer 类 ========================
class GATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        return ga_loss(model, inputs)

class NPOTrainer(Trainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model 
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        return npo_loss(model, self.ref_model, inputs)

class GDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return gd_npo_loss(model, self.ref_model, inputs)


class PenaltyDPOTrainer(Trainer):
    def __init__(
        self,
        ref_model,
        penalty_model: PenaltyDistillationModel,
        tokenizer,
        beta=0.5,
        alpha=0.2,
        penalty_strength=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.ref_model = ref_model
        self.penalty_model = penalty_model
        self.rapo_tokenizer = tokenizer
        self.tokenizer = tokenizer 
        self.beta = beta
        self.alpha = alpha
        self.penalty_strength = (
            penalty_strength
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        return penalty_based_dpo_loss(
            main_model=model,
            ref_model=self.ref_model,
            inputs=inputs,
            penalty_model=self.penalty_model,
            tokenizer=self.rapo_tokenizer,
            beta=self.beta,
            alpha=self.alpha,
            penalty_strength=(
                self.penalty_strength
            ),
            return_outputs=return_outputs,
        )

class PenaltyDOORTrainer(Trainer):
    def __init__(
        self,
        ref_model,
        penalty_model: PenaltyDistillationModel,
        tokenizer,
        beta=0.5,
        alpha=0.2,
        penalty_strength=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.ref_model = ref_model
        self.penalty_model = penalty_model
        self.rapo_tokenizer = tokenizer
        self.tokenizer = tokenizer 
        self.beta = beta
        self.alpha = alpha
        self.penalty_strength = (
            penalty_strength
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        return penalty_based_door_loss(
            main_model=model,
            ref_model=self.ref_model,
            inputs=inputs,
            penalty_model=self.penalty_model,
            tokenizer=self.rapo_tokenizer,
            beta=self.beta,
            alpha=self.alpha,
            penalty_strength=(
                self.penalty_strength
            ),
            return_outputs=return_outputs,
        )
        