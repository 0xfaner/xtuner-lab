"""E1 选做：对比微调前后的生成样例。"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "用一句话解释什么是梯度下降。",
    "给我三个保持健康的小建议。",
    "把这句话翻译成英文：今天天气真好，我们去公园散步吧。",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 格式模型目录（基座或训练产出的 hf-xxx）")
    ap.add_argument("--tokenizer", default=None, help="默认与 --model 相同；checkpoint 缺 tokenizer 时传基座目录")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    return ap.parse_args()


def load_model(path: str):
    kwargs = dict(device_map="cuda" if torch.cuda.is_available() else "cpu")
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kwargs)


def render_prompt(tokenizer, prompt: str):
    msgs = [{"role": "user", "content": prompt}]
    try:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = load_model(args.model)
    model.eval()

    for p in PROMPTS:
        ids = render_prompt(tokenizer, p).to(model.device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        print("=" * 60)
        print("问:", p)
        print("答:", answer.strip())


if __name__ == "__main__":
    main()
