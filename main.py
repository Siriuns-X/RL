import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
Q_ref = np.load("./data/FrozenLake_8x8_false.npy")
Q_prev = np.zeros((env.observation_space.n, env.action_space.n))
deltas_prev = []
deltas_ref = []

lr, gamma, eps = 0.8, 0.95, 1.0
n_ep = 2000
decay = 0.995
succ_bool = []
for ep in range(n_ep):
    Q_prev = Q.copy()
    s, _ = env.reset()
    done = False

    succ = False
    while not done:
        a = env.action_space.sample() if np.random.rand() < eps \
            else int(np.random.choice(np.flatnonzero(Q[s] == Q[s].max())))
        # a = env.action_space.sample() if np.random.rand() < eps else np.argmax(Q[s])
        s2, r, term, trunc, _ = env.step(a)
        Q[s, a] += lr * (r + gamma * np.max(Q[s2]) * (not term) - Q[s, a])
        s, done = s2, term or trunc
        succ = (r==1)

    succ_bool.append(1.0 if succ else 0.0)
    deltas_prev.append(np.abs(Q - Q_prev).max())
    deltas_ref.append(np.abs(Q - Q_ref).max())
    eps = max(0.05, eps * decay)

w = 200
succ_rate = np.convolve(succ_bool, np.ones(w)/w, mode="valid")
# 图2
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

ax1.plot(deltas_prev, ls="none", marker=".", ms=2)
ax1.set_ylim(1e-20, 2)
ax1.set_yscale("log")
# ax1.set_xscale("log")
ax1.set_ylabel("max |Q - Q_prev|")
ax1.set_title("prev")
ax1.grid(alpha=0.3)

ax2.plot(deltas_ref, lw=0.8, color="tab:orange")
ax2.axhline(0, ls="--", c="gray", lw=0.8)
# ax2.set_xscale("")
# ax2.set_yscale("")
ax2.set_xlabel("episode")
ax2.set_ylabel("Q - Q_ref")
ax2.set_title("ref")
ax2.grid(alpha=0.3)

ax3.plot(succ_rate, lw=0.8, color="tab:red")
ax3.axhline(0, ls="--", c="gray", lw=0.8)
ax3.grid(alpha=0.3)
ax3.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()