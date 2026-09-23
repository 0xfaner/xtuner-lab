"""双进程复现：CUDA 张量跨进程共享（A 默认 / B expandable_segments=True / C 新旧变量名覆盖）。"""
import os
import pickle
import subprocess
import sys
import time

import torch
import torch.multiprocessing.reductions as R

R.init_reductions()

ROLE = sys.argv[1] if len(sys.argv) > 1 else "orchestrate"
PKL = "/tmp/share_test.pkl"


def produce():
    t = torch.arange(16, dtype=torch.float32, device="cuda").reshape(2, 8)
    item = R.reduce_tensor(t)
    _, args = item
    h = args[7]
    hl = len(h) if isinstance(h, bytes) else h
    print(f"produce: handle={type(h).__name__} {hl}", flush=True)
    with open(PKL, "wb") as f:
        pickle.dump(item, f)
    print("produce: pickled, keep-alive 30s ...", flush=True)
    time.sleep(30)


def consume():
    time.sleep(3)
    item = pickle.load(open(PKL, "rb"))
    func, args = item
    args = list(args)
    args[6] = torch.cuda.current_device()
    t2 = func(*args)
    print("consume: OK,", t2.flatten()[:4].tolist(), flush=True)


if ROLE == "produce":
    produce()
elif ROLE == "consume":
    consume()
else:
    env = dict(os.environ)
    p = subprocess.Popen([sys.executable, __file__, "produce"], env=env)
    time.sleep(3)
    c = subprocess.run([sys.executable, __file__, "consume"], env=env, capture_output=True, text=True)
    p.kill()
    out = (c.stdout or "") + (c.stderr or "")
    tail = out.strip().splitlines()[-6:]
    verdict = "OK ✅" if c.returncode == 0 and "consume: OK" in (c.stdout or "") else "FAIL ❌"
    print(f"==== 结果: {verdict} (rc={c.returncode}) ====")
    print("\n".join(tail))
