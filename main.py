import gymnasium as gym, numpy as np
import numpy as np

import torch, torch.nn as nn
from collections import deque
import random
import copy

def sample_batch(buffer, batch_size, device="cpu"):
    s, a, r, s2, term, trunc = zip(*random.sample(buffer, batch_size))
    T = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
    return (T(np.stack(s), torch.float32), T(a, torch.int64), T(r, torch.float32),
            T(np.stack(s2), torch.float32), T(term, torch.bool))

def q_of(net, states, actions):
    return net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

def train_step(net, target_net, opt, buffer, batch_size, gamma):
    if len(buffer) < batch_size:
        return None
    s, a, r, s2, term = sample_batch(buffer, batch_size)
    with torch.no_grad():
        target = r + gamma * target_net(s2).max(1).values * (~ term)
    loss = nn.functional.mse_loss(q_of(net, s, a), target)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

seed = 42

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=seed)
env.action_space.seed(seed)

torch.manual_seed(seed)
net = nn.Sequential(nn.Linear(4,64), nn.ReLU(), nn.Linear(64,2))
target_net = copy.deepcopy(net)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

eps_start, eps_end, eps_decay_steps = 1.0, 0.05, 10_000
eps = 1.0
step_cnt = 0
n = 300
total = 0
vec = []
losses = []
q_mean = []
batch_size = 64
gamma = 0.99
buffer = deque(maxlen=10_100)

for i in range(n):
    obs, _ = env.reset()
    done = False
    cnt = 0
    while not done:
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - step_cnt / eps_decay_steps)
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
        train_step(net, target_net, opt, buffer, batch_size, gamma)

        step_cnt += 1
        if step_cnt % 500 == 0:
            target_net.load_state_dict(net.state_dict()) 

    total += cnt
    vec.append(cnt)