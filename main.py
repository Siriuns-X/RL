import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from collections import deque
import random

plt.rcParams.update({"font.size": 14, "figure.dpi": 110})

seed = 42

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=seed)
env.action_space.seed(seed)

torch.manual_seed(seed)
net = nn.Sequential(nn.Linear(4,64), nn.ReLU(), nn.Linear(64,2))

eps = 1.0
n = 300
sum = 0
vec = []
buffer = deque(maxlen=10_100)
for i in range(n):
    obs, _ = env.reset()
    done = False
    cnt = 0
    while not done:
        q_val = net(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ) # 神秘batch, 为了防止后续argmax处dim不一致, 所以加上unsqueeze
        act = env.action_space.sample() if np.random.rand() < eps \
            else q_val.argmax(dim=1).item()
        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        buffer.append((obs, act, reward, next_obs, terminated, truncated))
        cnt += 1
        obs = next_obs
    sum += cnt
    vec.append(cnt)

np.savez("./data/init.npz", vec=vec, seed=seed)

fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharey=True, sharex=True)
# ax = axes[0]
axes.plot([sum/n for _ in range(n)])
axes.plot(vec)
fig.savefig('./figs/init.png', dpi=300, bbox_inches='tight')
plt.show()