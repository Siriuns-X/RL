import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import lfilter

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

def ema(v, span):
    a = 2.0 / (span + 1.0)
    return lfilter([a], [1, -(1-a)], v, zi=[v[0]*(1-a)])[0]


ts = "20260818_003317"
run_dir = Path("runs") / ts
npz = np.load(run_dir / "curves.npz")

vales = ["losses", "vec", "evaluate_vec"]

q_vec = np.asarray(npz["q_vec"])
x, q_max, q_min = q_vec.T
q_min_clean = median_filter(q_min, size=10)
q_max_clean = median_filter(q_max, size=10)
q_min_smooth = ema(q_min_clean, span=1000)
q_max_smooth = ema(q_max_clean, span=1000)
delta = median_filter(q_max - q_min, size=10)
delta_smooth = ema(delta, span=1000)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x, q_min_smooth, label="q_min")
ax.plot(x, q_max_smooth, label="q_max")
fig.savefig(
    run_dir / f"q.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x, delta_smooth)
fig.savefig(
    run_dir / f"delta_q.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

for vale in vales:
    fig, ax = plt.subplots(figsize=(10, 7))
    x, y = zip(*npz[vale])
    ax.plot(x, y)
    fig.savefig(
        run_dir / f"{vale}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)