import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

ts = "20260815_002859"
run_dir = Path("runs") / ts
npz = np.load(run_dir / "curves.npz")


vales = ["q_mean", "losses", "vec"]

for vale in vales:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(npz[vale])
    fig.savefig(
        run_dir / f"{vale}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)