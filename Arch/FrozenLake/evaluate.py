import gymnasium as gym, numpy as np
import numpy as np
import matplotlib.pyplot as plt

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
Q = np.load("./data/FrozenLake_8x8_true.npy")

print(evaluate(env, Q, 10000))
