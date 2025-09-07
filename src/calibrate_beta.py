"""
Illustrative calibration of beta: choose beta to match average K/Y ~ target.
Note: For speed, this does NOT re-solve VFI for each beta; it uses the existing policy.
In a full calibration you would re-solve the model per beta.
"""
import numpy as np

dat = np.load("rbc_policy_data.npz")
k_grid = dat["k_grid"]; z_grid = dat["z_grid"]; policy_idx = dat["policy_idx"]
Nk, Nz = policy_idx.shape

alpha = 0.33
delta = 0.08

def simulate(beta, T=200, seed=0):
    rng = np.random.default_rng(seed)
    k = 1.0
    iz = Nz//2
    K = []; Y = []
    for t in range(T):
        ik = int(np.clip(np.searchsorted(k_grid, k), 0, Nk-1))
        kp = k_grid[policy_idx[ik, iz]]
        z = z_grid[iz]
        y = z * (k**alpha)
        K.append(k); Y.append(y)
        k = kp
        iz = rng.integers(0, Nz)
    K = np.array(K); Y = np.array(Y)
    ky = (K / (Y + 1e-8)).mean()
    return ky

target_KY = 2.5
beta_grid = np.linspace(0.92, 0.99, 15)
errs = []
for b in beta_grid:
    ky = simulate(b)
    errs.append((b, abs(ky - target_KY)))

best = min(errs, key=lambda t: t[1])
print("Target K/Y:", target_KY)
print("Best beta (coarse search):", best[0], " | abs error:", best[1])
