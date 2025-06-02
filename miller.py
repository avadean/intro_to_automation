import matplotlib.pyplot as plt
import numpy as np

A = 2.2
kappa = 1.5
delta = 0.3
R0 = 2.5
theta = np.linspace(0, 2 * np.pi)
r = R0 / A
R_s = R0 + r * np.cos(theta + (np.arcsin(delta) * np.sin(theta)))
Z_s = kappa * r * np.sin(theta)

plt.plot(R_s, Z_s)
plt.axis("equal")
plt.xlabel("R [m]")
plt.ylabel("Z [m]")
plt.savefig("../miller.png")
