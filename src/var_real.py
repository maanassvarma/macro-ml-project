"""
VAR(1) on real macro data from FRED (Real GDP & Real Personal Consumption Expenditures).
Requires internet and pandas_datareader. Saves IRF plot to docs/var_irfs.png

Note: In production you'd align frequencies, log-transform, and detrend carefully. Here we:
- Pull quarterly series GDPC1 (Real GDP) and PCEC96 (Real Personal Consumption Expenditures).
- Log-transform and difference once to approximate stationarity.
- Fit a VAR(1) and compute 12-step IRFs from a one-time GDP shock.
"""
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

start = "1990-01-01"
end = None

gdp = pdr.DataReader("GDPC1", "fred", start, end)
cons = pdr.DataReader("PCEC96", "fred", start, end)
df = pd.concat([gdp, cons], axis=1).dropna()
df.columns = ["gdp", "cons"]

# log difference
d = np.log(df).diff().dropna()

# VAR(1): Y_t = A Y_{t-1} + c + eps
Y = d.values[1:, :]                                    # (T-1) x 2
X = np.column_stack([d.values[:-1, :], np.ones(len(d)-1)])  # (T-1) x 3
B = np.linalg.pinv(X) @ Y                              # 3 x 2
A = B[:2, :]                                           # 2 x 2
c = B[2:3, :]                                          # 1 x 2

# IRFs for 12 periods from GDP unit shock
h = 12
Phi = [np.eye(2), A.copy()]
for _ in range(2, h+1):
    Phi.append(Phi[-1] @ A)
Phi = np.stack(Phi, axis=0)                           # (h+1) x 2 x 2

shock = np.array([1.0, 0.0])                          # GDP shock
irf = Phi @ shock                                     # (h+1) x 2

t = np.arange(h+1)
plt.figure()
plt.plot(t, irf[:,0], label="GDP response")
plt.plot(t, irf[:,1], label="Consumption response")
plt.axhline(0, linewidth=1)
plt.xlabel("horizon")
plt.ylabel("log-diff response")
plt.title("VAR(1) IRFs: GDP shock")
plt.legend()
plt.tight_layout()
plt.savefig("docs/var_irfs.png", dpi=140)
print("Saved docs/var_irfs.png")
