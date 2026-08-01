import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

def solve(env, gamma=0.95, iters=2000):
    P = env.unwrapped.P              # P[s][a] = [(prob, s2, r, done), ...]
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    for _ in range(iters):
        V = Q.max(axis=1)
        for s in range(nS):
            for a in range(nA):
                Q[s, a] = sum(p * (r + gamma * V[s2] * (not d)) for p, s2, r, d in P[s][a])
    return Q

Q = solve(env, 0.95, 2000)
np.save("./data/FrozenLake_8x8_true.npy", Q)