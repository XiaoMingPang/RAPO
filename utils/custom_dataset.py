import copy
from torch.utils.data import Dataset
import torch
import json

SYSTEM_PROMPT = ""


class CustomDataset(Dataset):
    def __init__(self, tokenizer, train_dataset_dict, max_words=30, pad=True):
        self.ann = {}
        self.prefix = {}
        print("dataset path:", train_dataset_dict)
        self.length = -1
        for key, value in train_dataset_dict.items():
            ann = []
            path, prefix = value
            with open(path, 'r') as f:
                if prefix:
                    for line in f:
                        if line.strip():
                            a = json.loads(line)["messages"]
                            assert len(a) == 3 and a[0]["role"] == "user" and a[1]["role"] == "prefix" and a[2]["role"] == "assistant"
                            ann.append({"user": a[0]["content"], "prefix": a[1]["content"], "assistant": a[2]["content"]})
                else:
                    for line in f:
                        if line.strip():
                            a = json.loads(line)["messages"]
                            assert len(a) == 2 and a[0]["role"] == "user" and a[1]["role"] == "assistant"
                            ann.append({"user": a[0]["content"], "assistant": a[1]["content"]})
            self.ann[key] = ann
            self.prefix[key] = prefix
            if self.length == -1:
                self.length = len(ann)
            else:
                assert self.length == len(ann)

        self.sys_role = False
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        try:
            tokenizer.apply_chat_template(chat)
        except:
            self.sys_role = False
        print("sys_prompt:", self.sys_role)

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        datasets = {}

        for key in self.ann.keys():
            ann = self.ann[key][index]
            prefix = self.prefix[key]

            if prefix:
                # ============================================================
                # [RAPO-FIX1] 前缀进 prompt、label 只含续写
                #   - 与 gen_response.py 的 prefill 推理严格对齐（推理时前缀是
                #     外部强制追加在 prompt 之后，模型只续写）。
                #   - 好处1：decode_rapo_responses 给惩罚模型喂的文本不再含
                #     有害前缀，规则命中不会集中在攻击者前缀上，risk_gap 不再
                #     被压成 0。
                #   - 好处2：DPO/DOOR 训练时 good/bad 共享同一 prompt（同一
                #     前缀），log-ratio 只反映续写差异，语义更干净。
                # ============================================================
                if self.sys_role:
                    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ann["user"]}]
                else:
                    chat = [{"role": "user", "content": ann["user"]}]

                # 用户 prompt + assistant 起始标记（与推理时一致）
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=True, add_generation_prompt=True)
                # 攻击者强制前缀：token 级追加到 prompt。
                # （prefix-bad 数据本为 prefix_ids + suffix_ids 的 round-trip，
                #  token 级拼接可避免字符串再切分的跨 token 边界问题）
                prefix_ids = self.tokenizer(
                    ann["prefix"], add_special_tokens=False)["input_ids"]
                prompt_ids = prompt + prefix_ids
                # 续写作为唯一 label 区域（label 只含续写，惩罚模型天然看不到前缀）
                cont_ids = self.tokenizer(
                    ann["assistant"], add_special_tokens=False)["input_ids"]
                eos_id = self.tokenizer.eos_token_id
                example_ids = prompt_ids + cont_ids + (
                    [eos_id] if eos_id is not None else [])

                prompt = torch.tensor(prompt_ids, dtype=torch.int64)
                example = torch.tensor(example_ids, dtype=torch.int64)
            else:
                if self.sys_role:
                    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ann["user"]}]
                else:
                    chat = [{"role": "user", "content": ann["user"]}]
                prompt = self.tokenizer.apply_chat_template(chat)
                chat.append({"role": "assistant", "content": ann["assistant"]})
                example = self.tokenizer.apply_chat_template(chat)
                prompt = torch.tensor(prompt, dtype=torch.int64)
                example = torch.tensor(example, dtype=torch.int64)
                # print(self.tokenizer.eos_token_id)
                # print(prompt, example)
                # print(self.tokenizer.decode(example))

            if self.pad:
                padding = self.max_words - example.shape[0]
                if padding > 0:
                    example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
                elif padding < 0:
                    example = example[: self.max_words]

            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = IGNORE_INDEX
            example_mask = example_mask.float()
            label_mask = label_mask.float()

            datasets.update(
                {
                    f"{key}input_ids": example,
                    f"{key}labels": labels,
                    f"{key}attention_mask": example_mask,
                    # [RAPO-FIX2] 导出原始 user 文本，供惩罚模型打分时带上问题上下文
                    f"{key}user": ann["user"],
                }
            )

        return datasets
