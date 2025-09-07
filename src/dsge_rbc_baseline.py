"""
RBC (Stochastic Growth) — small nonlinear DSGE via Value Function Iteration (VFI).
Exports:
- rbc_policy_data.npz (k_grid, z_grid, policy_idx)
- sim_series.csv (t,y,c,k,k_next,z_index)
- irf.csv (t, y_irf, c_irf)
Prints: convergence stats and simple IRFs.
"""
import numpy as np
import time
import csv

# Parameters (tiny instance for speed)
alpha = 0.33
beta = 0.96
delta = 0.08
rho = 0.9
sigma_eps = 0.02

# Grids
Nk = 120
Nz = 7
k_min, k_max = 0.5, 3.0
k_grid = np.linspace(k_min, k_max, Nk)

# Tauchen discretization for z
def tauchen(N, mu, rho, sigma, m=3):
    """
    Discretize AR(1): x' = mu + rho x + eps, eps ~ N(0, sigma^2)
    Returns grid x, transition matrix P.
    """
    std_x = np.sqrt(sigma**2 / (1 - rho**2))
    x_max = mu + m*std_x
    x_min = mu - m*std_x
    x = np.linspace(x_min, x_max, N)
    step = (x_max - x_min)/(N - 1)
    P = np.zeros((N, N))
    from math import erf, sqrt
    def Phi(x): return 0.5*(1+erf(x/np.sqrt(2)))
    for i in range(N):
        for j in range(N):
            z_upper = (x[j] - mu - rho*x[i] + step/2)/sigma
            z_lower = (x[j] - mu - rho*x[i] - step/2)/sigma
            if j == 0:
                P[i,j] = Phi(z_upper)
            elif j == N-1:
                P[i,j] = 1 - Phi(z_lower)
            else:
                P[i,j] = Phi(z_upper) - Phi(z_lower)
    return np.exp(x), P  # return z = exp(x)

z_grid, Pz = tauchen(Nz, 0.0, rho, sigma_eps)

# Utility with feasibility check
def utility(c):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(c>0, np.log(c), -1e10)
    return out

# VFI
V = np.zeros((Nk, Nz))
policy_idx = np.zeros((Nk, Nz), dtype=int)
max_iter = 500
tol = 1e-5

def EV_from_V(V):
    return V @ Pz.T  # Nk x Nz

t0 = time.time()
for it in range(max_iter):
    V_new = np.empty_like(V)
    pol_new = np.empty_like(policy_idx)
    EV = EV_from_V(V)  # refresh
    for iz, z in enumerate(z_grid):
        y = z * (k_grid**alpha)
        for ik, k in enumerate(k_grid):
            # feasible k' grid s.t. c = y + (1-delta)k - k' > 0
            c = y[ik] + (1 - delta)*k - k_grid
            u = utility(c)
            rhs = u + beta * EV[:, iz]
            j = np.argmax(rhs)
            V_new[ik, iz] = rhs[j]
            pol_new[ik, iz] = j
    diff = np.max(np.abs(V_new - V))
    V = V_new
    policy_idx = pol_new
    if diff < tol:
        break
t1 = time.time()
print(f"Converged in {it+1} iterations, sup-norm diff={diff:.2e}")
print(f"VFI time: {t1 - t0:.2f} s for Nk={Nk}, Nz={Nz}")

# Export policy dataset (for NN training later)
np.savez_compressed("rbc_policy_data.npz", k_grid=k_grid, z_grid=z_grid, policy_idx=policy_idx)

# Simulate T=200
T=200
k_series = np.empty(T+1)
z_idx_series = np.empty(T+1, dtype=int)
y_series = np.empty(T)
c_series = np.empty(T)

k_series[0] = 1.0
z_idx_series[0] = Nz//2

rng = np.random.default_rng(0)
for t in range(T):
    iz = z_idx_series[t]
    ik = int(np.clip(np.searchsorted(k_grid, k_series[t]), 0, Nk-1))
    kp_idx = policy_idx[ik, iz]
    kp = k_grid[kp_idx]
    z = z_grid[iz]
    y = z * (k_series[t]**alpha)
    c = y + (1-delta)*k_series[t] - kp
    y_series[t] = y
    c_series[t] = c
    k_series[t+1] = kp
    # draw z_{t+1}
    z_idx_series[t+1] = rng.choice(Nz, p=Pz[iz])

print("Simulated series summary:")
print(" y mean/std:", y_series.mean(), y_series.std())
print(" c mean/std:", c_series.mean(), c_series.std())

# Save simulation CSV
import csv
with open("sim_series.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","y","c","k","k_next","z_index"])
    for t in range(T):
        w.writerow([t, y_series[t], c_series[t], k_series[t], k_series[t+1], int(z_idx_series[t])])

# Simple one-shock IRF (deterministic z transition)
def irf_shock(shock_steps=40, shock_mag=2.0):
    k = 1.0
    iz = Nz//2
    irf_y, irf_c = [], []
    for t in range(shock_steps):
        z = z_grid[iz]
        if t == 1:
            z *= shock_mag
        ik = int(np.clip(np.searchsorted(k_grid, k), 0, Nk-1))
        kp_idx = policy_idx[ik, iz]
        kp = k_grid[kp_idx]
        y = z * (k**alpha)
        c = y + (1-delta)*k - kp
        irf_y.append(y); irf_c.append(c)
        k = kp
        iz = int(np.argmax(Pz[iz]))
    return np.array(irf_y), np.array(irf_c)

iy, ic = irf_shock()
print("IRF (first 5 y):", np.round(iy[:5], 4))
print("IRF (first 5 c):", np.round(ic[:5], 4))

# Save IRF CSV
with open("irf.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","y_irf","c_irf"])
    for t in range(len(iy)):
        w.writerow([t, iy[t], ic[t]])
