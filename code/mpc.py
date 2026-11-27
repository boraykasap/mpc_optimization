import numpy as np

# ---------------- Q2: prediction matrices ----------------
def prediction_matrices(T):
    I2 = np.eye(2)
    A = np.zeros((4 * T, 2 * T))
    C = np.zeros((4 * T, 4))
    for k in range(1, T + 1):
        rows = np.arange(4 * (k - 1), 4 * k)
        C[rows, :] = np.kron(np.array([[1, k], [0, 1]]), I2)
        for j in range(1, k + 1):
            m = k - j
            A[rows, 2 * (j - 1):2 * j] = np.kron(np.array([[m], [1]]), I2)
    return A, C

# ---------------- Q3: unconstrained MPC ----------------
def mpcfree(z_t, T, A=None, C=None):
    if A is None or C is None:
        A, C = prediction_matrices(T)
    u_star = np.linalg.solve(A.T @ A, -A.T @ (C @ z_t))
    return u_star[:2]