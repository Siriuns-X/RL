import gymnasium as gym, numpy as np
import numpy as np
import time
import json

def evaluate(env, Q, n=1000):
    wins = 0
    for _ in range(n):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            done = term or trunc
        wins += (r > 0)
    return wins / n

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
Q = np.zeros((env.observation_space.n, env.action_space.n))
Q_ref = np.load("./data/FrozenLake_8x8_true.npy")
Q_prev = np.zeros((env.observation_space.n, env.action_space.n))
deltas_prev = []
deltas_ref = []

lr, gamma, eps, n_ep = 0.20, 0.90, 1.0, 50000
decay = (0.05 / 1.0) ** (1 / (n_ep * 0.5))

succ_bool = []
eval_points = np.unique(np.geomspace(10, n_ep, 100).astype(int))
acc_his = []
vis = np.zeros_like(Q)

for ep in range(n_ep):
    Q_prev = Q.copy()
    s, _ = env.reset()
    done = False

    succ = False
    while not done:
        a = env.action_space.sample() if np.random.rand() < eps \
            else int(np.random.choice(np.flatnonzero(Q[s] == Q[s].max())))
        s2, r, term, trunc, _ = env.step(a)
        vis[s, a] += 1
        ls_sa = 1.0 / vis[s, a] ** 0.7
        Q[s, a] += ls_sa * (r + gamma * np.max(Q[s2]) * (not term) - Q[s, a])
        
        s, done = s2, term or trunc
        succ = (r==1)

    succ_bool.append(1.0 if succ else 0.0)
    deltas_prev.append(np.abs(Q - Q_prev).max())
    deltas_ref.append(np.abs(Q - Q_ref).max())
    if ep in eval_points:
        acc_his.append((ep, evaluate(env, Q, n=500)))
    eps = max(0.05, eps * decay)

w = 200
succ_rate = np.convolve(succ_bool, np.ones(w)/w, mode="valid")

np.savez(f"./runs/8x8_slip_{int(time.time())}_{n_ep}.npz",
         Q=Q, succ_bool=succ_bool, delta=deltas_prev,
         acc_his=acc_his, succ_rate=succ_rate,
         meta=json.dumps({"lr": lr, "gamma": gamma, "n_ep": n_ep, "decay": decay}))