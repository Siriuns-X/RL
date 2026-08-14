import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

npz = np.load("./data/20260814_231011.npz")

fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharey=True, sharex=True)
# ax = axes[0]
vale = "vec"
axes.plot(npz[vale])
fig.savefig(f"./figs/{vale}_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
plt.show()