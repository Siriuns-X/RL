import gymnasium as gym, numpy as np
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import torch, torch.nn as nn
from collections import deque
import random
import copy
import subprocess

def new_run(hp, comment=""):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "diff", "--quiet"]).returncode)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("runs") / ts
    d.mkdir(parents=True)
    meta = {"ts": ts, "commit": commit, "dirty": dirty,
            "comment": comment, **hp}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with open("runs/index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return d

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

def evaluate(env, net, n=10):
    if n <= 0: return -1
    cnt = 0
    for i in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            q_val = net(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                )
            act = q_val.argmax(dim=1).item()
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            cnt += 1
            obs = next_obs
    return cnt/n

hp = dict(seed=42, n_ep=600, lr=1e-3, gamma=0.99, batch_size=64,
          use_target=True, target_sync=500,
          eps_start=1.0, eps_end=0.05,
          eps_decay_steps=10_000,
          buffer_size=100_000,
          )

run_dir = new_run(hp, comment="scale n_ep and buffer size")

seed = hp["seed"]

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=seed)
env.action_space.seed(seed)

torch.manual_seed(seed)
net = nn.Sequential(nn.Linear(4,64), nn.ReLU(), nn.Linear(64,2))
target_net = copy.deepcopy(net)
opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])

eps_start, eps_end, eps_decay_steps = hp["eps_start"], hp["eps_end"], hp["eps_decay_steps"]

step_cnt = 0
total = 0
vec = []
evaluate_vec = []
losses = []
q_mean = []
buffer = deque(maxlen=hp["buffer_size"])


for ep in range(hp["n_ep"]):
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
        l = train_step(net, target_net, opt, buffer, hp["batch_size"], hp["gamma"])
        if l is not None: losses.append(l)
        q_mean.append(q_val.max().item())

        step_cnt += 1
        if step_cnt % hp["target_sync"] == 0:
            target_net.load_state_dict(net.state_dict()) 
    if ep % 20 == 0:
        evaluate_vec.append(evaluate(env, net, n=10))
    total += cnt
    vec.append(cnt)


np.savez(run_dir / "curves.npz", vec=vec, losses=losses, q_mean=q_mean, total=total, evaluate_vec=evaluate_vec)
torch.save(net.state_dict(), run_dir / "net.pt")
print(run_dir)