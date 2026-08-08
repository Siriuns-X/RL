import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from scipy.signal import savgol_filter

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

meta = []
acc = []


folder = Path("./runs")
files = sorted(folder.glob("*.npz"))

for file in files:
    with np.load(file) as npz:
        acc.append(npz["acc_his"])
        meta.append(json.loads(str(npz["meta"])))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)

for ax, m, a in zip(axes.flat, meta, acc):
    a = np.asarray(a)
    x, y = a[:, 0], a[:, 1]
    
    sm = savgol_filter(y, window_length=21, polyorder=2)
    ax.plot(x, y, alpha=0.9, lw=0.6, color="C0")
    ax.set_title(f"lr={m['lr']}  gamma={m['gamma']}", fontsize=12)
    ax.plot(x, sm, lw=1.5, color="red")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)

fig.savefig("figs/8x8_slip_hp_sweep.pdf", bbox_inches="tight")
fig.savefig("figs/8x8_slip_hp_sweep.png", dpi=150, bbox_inches="tight")