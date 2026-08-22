import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
from glob import glob
import random
import numpy as np

def load_model_and_tokenizer(model_path, dtype=torch.bfloat16, max_memory=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = dict(
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer

def load_conversations(eval_path):
    if eval_path.endswith(".jsonl"):
        with open(eval_path, 'r') as file:
            json_lines = file.readlines()
        conversations =  [
            [{"role": "user", "content": json.loads(line)['input']}]
            for line in json_lines
        ]
    elif eval_path.endswith(".csv"):
        df = pd.read_csv(eval_path)
        conversations =  [
            [{"role": "user", "content": row['prompt']}]
            for _, row in df.iterrows()
        ]
    else:
        raise ValueError("Unsupported file format. Please provide a JSONL or CSV file.")
    return conversations

def load_multi_turn_conversations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        all_conversations = [
            {
                "objective": entry.get("objective", ""),
                "conversation": entry.get("conversation", [])
            }
            for entry in data if "conversation" in entry
        ]
    return all_conversations

def load_harmbench_conversations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        all_conversations = []
        for objective, prompts in data.items():
            for prompt in prompts:
                all_conversations.append({
                    "objective": objective,
                    "prompt": prompt
                })
    return all_conversations



def _safe_truncation(tokenizer, side):
    """临时设置 truncation_side，不污染全局"""
    old = tokenizer.truncation_side
    tokenizer.truncation_side = side
    return old


def _restore_truncation(tokenizer, old):
    tokenizer.truncation_side = old


def _build_inputs(tokenizer, formatted_inputs, model_max_length, max_new_tokens,
                  truncation_side="left"):
    old = _safe_truncation(tokenizer, truncation_side)
    try:
        inputs = tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_max_length - max_new_tokens,
        ).to("cuda")
    finally:
        _restore_truncation(tokenizer, old)
    return inputs


def _slice_outputs(outputs, input_ids):
    """按 token 切，更鲁棒（不再依赖字符级边界）"""
    gen_ids = outputs[:, input_ids.shape[1]:]
    return gen_ids



def generate_general_responses(model, tokenizer, conversations, args):
#    all_inputs = [conv[0]['content'] for conv in conversations]
#    all_decoded_outputs = []
#    all_decoded_inputs = []
    all_inputs_text = [conv[0]["content"] for conv in conversations]
    all_outputs_text = []
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    max_new_tokens = 512
    for i in tqdm(range(0, len(conversations), args.batch_size), desc="Generating General Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        
        try:
            formatted = [
                tokenizer.apply_chat_template(conv, tokenize=False,
                                             add_generation_prompt=True)
                for conv in batch_convs
            ]
            inputs = _build_inputs(tokenizer, formatted, model_max_length,
                                   max_new_tokens, truncation_side="left")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = _slice_outputs(outputs, inputs["input_ids"])
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_outputs_text.extend(decoded)
        except Exception as e:
            print(f"[WARN] batch {i} failed: {e}; fallback to empty")
            all_outputs_text.extend([""] * len(batch_convs))

    return [
        {"intent": all_inputs_text[i], "response_generated": all_outputs_text[i]}
        for i in range(len(all_inputs_text))
    ]

def generate_multi_turn_responses(model, tokenizer, conversations, args):
    all_objectives = [c["objective"] for c in conversations]
    all_outputs_text = []

    model_max_length = getattr(model.config, "max_position_embeddings", 4096)
    max_new_tokens = 512

    for i in tqdm(range(0, len(conversations), args.batch_size),
                  desc="Generating Multi-Turn Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        try:
            formatted = [
                tokenizer.apply_chat_template(c["conversation"], tokenize=False,
                                             add_generation_prompt=True)
                for c in batch_convs
            ]
            inputs = _build_inputs(tokenizer, formatted, model_max_length,
                                   max_new_tokens, truncation_side="left")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = _slice_outputs(outputs, inputs["input_ids"])
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_outputs_text.extend(decoded)
        except Exception as e:
            print(f"[WARN] multi-turn batch {i} failed: {e}; fallback to empty")
            all_outputs_text.extend([""] * len(batch_convs))

    return [
        {
            "objective": conversations[i]["objective"],
            "conversation": conversations[i]["conversation"],
            "response_generated": all_outputs_text[i],
        }
        for i in range(len(conversations))
    ]

def generate_harmbench_responses(model, tokenizer, conversations, args):
    output_data = []
    model_max_length = getattr(model.config, "max_position_embeddings", 4096)
    max_new_tokens = 256

    for i in tqdm(range(0, len(conversations), args.batch_size),
                  desc="Generating Harmbench Responses"):
        batch_convs = conversations[i:i + args.batch_size]
        try:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": c["prompt"]}],
                    tokenize=False, add_generation_prompt=True
                )
                for c in batch_convs
            ]
            inputs = _build_inputs(tokenizer, formatted, model_max_length,
                                   max_new_tokens, truncation_side="left")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = _slice_outputs(outputs, inputs["input_ids"])
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            for j, c in enumerate(batch_convs):
                output_data.append({
                    "objective": c["objective"],
                    "prompt": c["prompt"],
                    "response_generated": decoded[j],
                })
        except Exception as e:
            print(f"[WARN] harmbench batch {i} failed: {e}; fallback to empty")
            for c in batch_convs:
                output_data.append({
                    "objective": c["objective"],
                    "prompt": c["prompt"],
                    "response_generated": "",
                })

    return output_data

def save_output(output_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
def save_name_for(model_path):
    save_name = os.path.basename(model_path) or "model"
    former_dir = os.path.basename(os.path.dirname(model_path)) or "exp"
    return f"{former_dir}_{save_name}"

def process_general_responses(model, tokenizer, args):
    conversations = load_conversations(args.eval_path)
    output_data = generate_general_responses(model, tokenizer, conversations, args)
    output_file = os.path.join(
        args.output_dir, f"{save_name_for(args.model_path)}_oversafe.json",
    )
    save_output(output_data, output_file)
    print(f"General responses saved to {output_file}")

def load_prefill_data(prefill_path):
    with open(prefill_path, 'r') as file:
        input_data = [json.loads(line) for line in file]
    return input_data

def generate_prefill_responses(model, tokenizer, input_data, args):
    save_name = os.path.basename(args.model_path)
    former_dir = os.path.basename(os.path.dirname(args.model_path))

    model_max_length = getattr(model.config, "max_position_embeddings", 4096)
    max_new_tokens = 256

    for prefill_len in args.prefill_lengths:
        print(f"Generating with prefill length: {prefill_len} tokens")
        all_decoded_outputs, all_prefills, all_inputs = [], [], []
        conversations = []

        for entry in input_data:
            question = entry["messages"][0]["content"]
            response = entry["messages"][1]["content"]
            response_tokens = tokenizer.tokenize(response)
            prefill_tokens = response_tokens[:prefill_len]
            prefill_text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(prefill_tokens),
                skip_special_tokens=True
            )
            if not prefill_text.strip():
                prefill_text = ""

            all_prefills.append(prefill_text)
            all_inputs.append(question)
            conv = [{"role": "user", "content": question}]
            new_input = (tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            ) + " " + prefill_text)
            conversations.append(new_input)

        for i in tqdm(range(0, len(conversations), args.batch_size),
                      desc=f"Processing Prefill Length {prefill_len}"):
            batch_convs = conversations[i:i + args.batch_size]
            try:
                inputs = _build_inputs(tokenizer, batch_convs, model_max_length,
                                       max_new_tokens, truncation_side="left")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                gen_ids = _slice_outputs(outputs, inputs["input_ids"])
                decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                all_decoded_outputs.extend(decoded)
            except Exception as e:
                print(f"[WARN] prefill batch {i} failed: {e}; fallback to empty")
                all_decoded_outputs.extend([""] * len(batch_convs))

        save_data = [
            {
                "intent": all_inputs[i],
                "number_of_words_prefilled": prefill_len,
                "harmful_prefix": all_prefills[i],
                "response_generated": all_decoded_outputs[i],
            }
            for i in range(len(all_inputs))
        ]

        output_file = os.path.join(
            args.output_dir, f"{former_dir}_{save_name}_prefill_{prefill_len}.json"
        )
        save_output(save_data, output_file)
        print(f"Prefill responses with {prefill_len} words saved to {output_file}")


def process_prefill_responses(model, tokenizer, args):
    input_data = load_prefill_data(args.eval_path)
    generate_prefill_responses(model, tokenizer, input_data, args)

def process_multi_turn_responses(model, tokenizer, args):
    conversations = load_multi_turn_conversations(args.eval_path)
    output_data = generate_multi_turn_responses(model, tokenizer, conversations, args)
    output_file = os.path.join(
        args.output_dir, f"{save_name_for(args.model_path)}_multi_turn.json"
    )
    save_output(output_data, output_file)
    print(f"Multi-turn responses saved to {output_file}")

def process_harmbench_responses(model, tokenizer, args):
    conversations = load_harmbench_conversations(args.eval_path)
    output_data = generate_harmbench_responses(model, tokenizer, conversations, args)
    output_file = os.path.join(
        args.output_dir, f"{save_name_for(args.model_path)}_harmbench.json"
    )
    save_output(output_data, output_file)
    print(f"Harmbench responses saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses using a language model.")
    parser.add_argument("--mode", choices=["general", "prefill", "multi_turn", "harmbench"], required=True, help="Mode")
    parser.add_argument("--model_path", required=True, help="Path or name of the pre-trained model")
    parser.add_argument("--eval_path", required=True, help="Path to input file (JSONL, CSV, or Harmbench JSON)")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--gpu", default="6", help="CUDA device to use")
    parser.add_argument("--prefill_lengths", type=int, nargs='*', default=[0, 3, 4, 5, 7, 10, 15, 20, 25], help="Word counts for prefill mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 设置所有随机种子（消融实验需要可复现）
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print("Model and tokenizer loaded successfully.")

    if args.mode == "general":
        process_general_responses(model, tokenizer, args)
    elif args.mode == "prefill":
        process_prefill_responses(model, tokenizer, args)
    elif args.mode == "multi_turn":
        process_multi_turn_responses(model, tokenizer, args)
    elif args.mode == "harmbench":
        process_harmbench_responses(model, tokenizer, args)
    else:
        raise ValueError("Invalid generation type. Choose from 'general', 'prefill', 'multi_turn', or 'harmbench'.")

if __name__ == "__main__":
    main()