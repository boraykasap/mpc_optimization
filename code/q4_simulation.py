import numpy as np
import matplotlib.pyplot as plt
from mpc import prediction_matrices, mpcfree

# ---------------- problem data ----------------
M = np.kron(np.array([[1, 1], [0, 1]]), np.eye(2))
B = np.kron(np.array([[0], [1]]), np.eye(2))
z0 = np.array([1.0, 0.0, 0.0, 0.25])
sigma = 0.01
T = 10
n_steps = 200
u_max = 0.01
rng = np.random.default_rng(0)

A, C = prediction_matrices(T)

# ---------------- simulation ----------------
Z = np.zeros((n_steps + 1, 4))
U = np.zeros((n_steps, 2))
Z[0] = z0
for t in range(n_steps):
    U[t] = mpcfree(Z[t], T, A, C)
    Z[t + 1] = M @ Z[t] + B @ U[t] + sigma * rng.standard_normal(4)

# ---------------- plots ----------------
t_z = np.arange(Z.shape[0])
t_u = np.arange(U.shape[0])

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(t_z, np.linalg.norm(Z, axis=1), color="#257de0", linewidth=1.5)
ax.set_xlabel("time step $t$")
ax.set_ylabel(r"$\|z_t\|$")
ax.grid(True, which="major", color="0.9", linewidth=0.8)
fig.tight_layout()
fig.savefig("results/q4_state_norm.png")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(t_u, np.max(np.abs(U), axis=1), color="#257de0", linewidth=1.5)
ax.axhline(u_max, color="0.35", linestyle="--", linewidth=1)
ax.text(1.01, u_max, r"$u_{\max}$", transform=ax.get_yaxis_transform(),
        ha="left", va="center", color="0.35")
ax.set_xlabel("time step $t$")
ax.set_ylabel(r"$\|u_t\|_\infty$")
ax.grid(True, which="major", color="0.9", linewidth=0.8)
fig.tight_layout()
fig.savefig("results/q4_controls.png")

plt.show()