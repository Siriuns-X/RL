import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt

# 一维数据显式加
x = torch.linspace(-3, 3, 500).unsqueeze(1)   # (500, 1) 列向量
y = torch.sin(x)

# 创建网络
net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
# optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = []
for step in range(2000):
    pred = net(x)
    loss = nn.functional.mse_loss(pred, y)

    opt.zero_grad()      # 清空上一轮梯度
    loss.backward()      # 反向传播，算 d(loss)/d(参数)
    opt.step()           # 按梯度更新参数

    losses.append(loss.item())

plt.plot(losses); plt.yscale("log"); plt.show()
plt.plot(x, y, label="true")
plt.plot(x, net(x).detach(), label="fit"); plt.legend(); plt.show()