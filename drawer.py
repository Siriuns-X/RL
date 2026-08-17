import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

ts = "20260817_190611"
run_dir = Path("runs") / ts
npz = np.load(run_dir / "curves.npz", allow_pickle=True)

vales = ["q_max", "q_min", "losses", "vec", "evaluate_vec"]

reasons = npz["reasons"].item()
print(reasons)
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