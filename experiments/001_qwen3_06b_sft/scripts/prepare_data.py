"""E1 数据准备：alpaca-gpt4-zh → train/val（OpenAI messages 格式，固定种子切分）。"""
import argparse
import json
import os
import random
import sys


def parse_args():
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.path.expanduser("~/workspace/xtuner-lab/data/alpaca_gpt4_data_zh.json"),
    )
    ap.add_argument("--out-dir", default=os.path.join(exp_dir, "data"))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.src):
        sys.exit(f"找不到原始数据文件：{args.src}（先下载 alpaca-gpt4-zh 到该路径）")

    with open(args.src, encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    skipped = 0
    for item in raw:
        instr = (item.get("instruction") or "").strip()
        extra = (item.get("input") or "").strip()
        out = (item.get("output") or "").strip()
        if not instr or not out:
            skipped += 1
            continue
        user = instr if not extra else f"{instr}\n{extra}"
        records.append(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": out},
            ]
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    train, val = records[: args.n_train], records[args.n_train : args.n_train + args.n_val]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[写出] {path}  {len(rows)} 条")

    print(f"\n原始 {len(raw)} 条 → 过滤后 {len(records)} 条（跳过 {skipped} 条空样本）")
    print(f"训练 {len(train)} 条 / 验证 {len(val)} 条；随机种子 {args.seed}")


if __name__ == "__main__":
    main()
