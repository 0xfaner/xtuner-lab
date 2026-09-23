import sys

import matplotlib.pyplot as plt
import pandas as pd

path, out = sys.argv[1], sys.argv[2]

df = pd.read_json(path, lines=True)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
axes[0].plot(df["step"], df["loss/reduced_llm_loss"])
axes[0].set_ylabel("loss")
axes[1].plot(df["step"], df["grad_norm"])
axes[1].set_ylabel("grad_norm")
axes[1].set_xlabel("step")
fig.tight_layout()

fig.savefig(out + ".png", dpi=150)
fig.savefig(out + ".svg")
print("saved:", out + ".png", "/", out + ".svg")
