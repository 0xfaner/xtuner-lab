"""验证集评估：assistant-only token 平均交叉熵（与训练口径一致）。"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.v1.data_proto.messages.chat import ChatMessages
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 格式模型目录（基座或训练产出的 hf-xxx）")
    ap.add_argument("--data", required=True, help="jsonl，每行一条 OpenAI messages 对话")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer 目录；默认与 --model 相同；checkpoint 里没有 tokenizer 文件时传基座目录",
    )
    return ap.parse_args()


def load_model(path: str):
    kwargs = dict(device_map="cuda" if torch.cuda.is_available() else "cpu")
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kwargs)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = load_model(args.model)
    model.eval()
    template = CHAT_TEMPLATE_MAP["qwen3"]

    total_nll, total_tokens, n = 0.0, 0, 0
    with torch.no_grad():
        for line in open(args.data, encoding="utf-8"):
            msgs = json.loads(line)
            out = ChatMessages(messages=msgs).tokenize(tokenizer, template)
            input_ids = torch.tensor([out["input_ids"]], device=model.device)
            labels = torch.tensor([out["labels"]], device=model.device)
            loss = model(input_ids=input_ids, labels=labels).loss
            n_valid = int((labels != -100).sum())
            total_nll += float(loss) * n_valid
            total_tokens += n_valid
            n += 1

    print(f"样本数: {n}")
    print(f"有效 token 数（只算 assistant 部分）: {total_tokens}")
    print(f"平均每 token 交叉熵: {total_nll / total_tokens:.4f}")


if __name__ == "__main__":
    main()
