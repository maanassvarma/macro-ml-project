"""
Train a tiny NumPy-based MLP to approximate the RBC policy function k' = f(k,z).
- Creates a held-out off-grid test set by jittering k
- Reports train and test RMSE
- Saves docs/results.png (policy slice) and docs/policy_error.png (error vs k)
"""
import numpy as np

data = np.load("rbc_policy_data.npz")
k_grid = data["k_grid"]
z_grid = data["z_grid"]
policy_idx = data["policy_idx"]  # Nk x Nz

# Build supervised dataset (X -> y), where X = (k, z), y = k'
X, y = [], []
for iz, z in enumerate(z_grid):
    for ik, k in enumerate(k_grid):
        kp = k_grid[policy_idx[ik, iz]]
        X.append([float(k), float(z)])
        y.append(float(kp))
X = np.array(X, dtype=float)
y = np.array(y, dtype=float).reshape(-1,1)

# Construct off-grid test by jittering k
rng = np.random.default_rng(1)
jitter = rng.normal(0, 0.01, size=len(X))
X_test = X.copy()
X_test[:,0] = np.clip(X_test[:,0] + jitter, k_grid.min(), k_grid.max())

# For test targets, interpolate k' from nearest 2 neighbors (simple linear interp over k at same z)
def interp_policy(x):
    k,z = x
    # find nearest z index
    iz = np.argmin(np.abs(z_grid - z))
    # find neighbors in k_grid
    ik = np.searchsorted(k_grid, k)
    if ik<=0: return float(k_grid[policy_idx[0,iz]])
    if ik>=len(k_grid): return float(k_grid[policy_idx[-1,iz]])
    k0,k1 = k_grid[ik-1], k_grid[ik]
    w = (k - k0)/(k1 - k0 + 1e-12)
    kp0 = k_grid[policy_idx[ik-1, iz]]
    kp1 = k_grid[policy_idx[ik, iz]]
    return float((1-w)*kp0 + w*kp1)

y_test = np.array([interp_policy(x) for x in X_test], dtype=float).reshape(-1,1)

# Normalize features
X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
Xn = (X - X_mean)/X_std
Xn_test = (X_test - X_mean)/X_std

# Tiny MLP: 2 -> 32 -> 1 with ReLU
rng = np.random.default_rng(0)
def init_layer(m,n):
    w = rng.normal(0, 1/np.sqrt(m), size=(m,n))
    b = np.zeros((1,n))
    return w,b

W1,b1 = init_layer(2,32)
W2,b2 = init_layer(32,1)

def relu(x): return np.maximum(0,x)
def forward(X):
    H = relu(X@W1 + b1)
    out = H@W2 + b2
    return out, H

def mse(a,b):
    d = a-b
    return float(np.mean(d*d))

lr = 1e-2
epochs = 500
for ep in range(epochs):
    out, H = forward(Xn)
    loss = mse(out, y)
    # Backprop
    dOut = 2*(out - y)/len(y)
    dW2 = H.T @ dOut
    db2 = dOut.sum(axis=0, keepdims=True)
    dH = dOut @ W2.T
    dH[H<=0] = 0
    dW1 = Xn.T @ dH
    db1 = dH.sum(axis=0, keepdims=True)
    # Update
    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1
    if (ep+1)%100==0:
        print(f"epoch {ep+1}: mse={loss:.6f}")

# Final evaluation
out_train,_ = forward(Xn)
rmse_train = np.sqrt(mse(out_train, y))

out_test,_ = forward(Xn_test)
rmse_test = np.sqrt(mse(out_test, y_test))

print(f"Policy NN RMSE (train): {rmse_train:.6f}")
print(f"Policy NN RMSE (test , off-grid): {rmse_test:.6f}")

# Save slice plot and error plot
try:
    import matplotlib.pyplot as plt
    iz = len(z_grid)//2
    ks = k_grid
    Xslice = np.column_stack([ks, np.full_like(ks, z_grid[iz])])
    Xs = (Xslice - X_mean)/X_std
    ys_pred, _ = forward(Xs)
    plt.figure()
    plt.plot(ks, ks[policy_idx[:, iz]], label="Optimal policy (grid)")
    plt.plot(ks, ys_pred[:,0], label="NN policy")
    plt.xlabel("k"); plt.ylabel("k'"); plt.title("Policy vs NN (z = mid state)")
    plt.legend(); plt.tight_layout()
    plt.savefig("docs/results.png", dpi=140)
    print("Saved docs/results.png")

    # error vs k on test set at mid z
    mask = np.isclose(X_test[:,1], z_grid[iz], atol=1e-6)
    if mask.any():
        import numpy as np
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(X_test[mask,0], (out_test[mask,0]-y_test[mask,0]), s=6)
        plt.axhline(0, linewidth=1)
        plt.xlabel("k"); plt.ylabel("NN error (k' units)")
        plt.title("Policy NN error vs k (off-grid, z ≈ mid)")
        plt.tight_layout()
        plt.savefig("docs/policy_error.png", dpi=140)
        print("Saved docs/policy_error.png")
except Exception as e:
    print("Plotting skipped:", e)

