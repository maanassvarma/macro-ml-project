
````markdown
# ML-Augmented Real Business Cycle (RBC) Model

Compact research-style repo for a nonlinear RBC (stochastic growth) model solved via Value Function Iteration (VFI), plus:
- **Compiled kernels** (C++ and Fortran) for the Bellman backup and timing,
- A tiny **NumPy MLP** surrogate to approximate the optimal policy \(k' = f(k, z)\),
- **Linear baseline** VAR(1) (simulated + real FRED data),
- A minimal **calibration** demo (β via a moment match),
- Reproducible scripts, unit test, and HPC-adjacent build flow.

## Motivation
Efficiently solving and experimenting with workhorse DSGE models is central to macro research. This repo shows clean implementations, performance-aware kernels, and a pragmatic ML surrogate—presented in a way economists can reproduce.

## Components
- **Python baseline (VFI):** `src/dsge_rbc_baseline.py`  
  Tauchen discretization, VFI on \((k,z)\), simulation, IRFs, CSV artifacts.
- **C++/Fortran kernels:** `src/dsge_rbc_cpp.cpp`, `src/dsge_rbc_fortran.f90`  
  Single-sweep Bellman backup; timing; optional OpenMP.
- **ML surrogate:** `src/dsge_policy_nn.py`  
  2→32→1 MLP (NumPy) trained on optimal policy; off-grid test + error plot.
- **VAR(1) baselines:** `src/var_linear.py` (toy), `src/var_real.py` (FRED GDP & Consumption).  
- **Calibration (illustrative):** `src/calibrate_beta.py` (coarse β fit to target K/Y).
- **Benchmarks:** `benchmarks/run_benchmarks.py` → `benchmarks/benchmarks.json`.
- **HPC bits:** `hpc/Makefile` (build C++/Fortran; `OMP=1` enables OpenMP), `hpc/run_slurm_example.sh`.
- **Test:** `tests/test_policy_monotonicity.py` enforces monotone policy in \(k\) at fixed \(z\).

## Getting Started
Python 3.9+ recommended.
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
````

## Reproduce

```bash
make reproduce        # VFI, NN, VARs, calibration; builds C++/Fortran; runs both
make bench            # writes benchmarks/benchmarks.json
make test             # unit test (policy monotonicity)
```

OpenMP (optional):

```bash
make -C hpc clean && make -C hpc OMP=1
```

## Results

**Nonlinear RBC (VFI).** Converged in \~230 iterations at sup-norm ≈ 1e-5. Example run:
*“Converged in 230 iterations, sup-norm diff=9.74e-06; VFI time ≈ 1.39 s for (Nk, Nz) = (120, 7).”*

**ML surrogate.** NumPy MLP policy approximator achieved train/test RMSE ≈ **0.131** on off-grid jittered samples.

* Policy slice (mid productivity state):
  **Put this image file at:** `docs/results.png`

* Off-grid error profile (mid state):
  **Put this image file at:** `docs/policy_error.png`

**Linear baseline (real data).** VAR(1) on FRED GDPC1 & PCEC96 (log-diff), 12-step IRFs from a GDP shock:
**Put this image file at:** `docs/var_irfs.png`

### Benchmarks (this machine)

From `make bench` and kernel timers:

| Kernel                 | Size (Nk,Nz) | Time (s)        | Notes             |
| ---------------------- | ------------ | --------------- | ----------------- |
| Python backup (median) | 120,7        | **0.066224**    | inline kernel     |
| C++ per-sweep          | 120,7        | **0.000441888** | `bellman_demo`    |
| Fortran per-sweep      | 120,7        | **0.000398865** | `bellman_fortran` |

**Test.** `pytest` passes: policy $k'$ is non-decreasing in $k$ at fixed $z$.

## Notes & Limitations

* Calibration script is **illustrative** (does not re-solve VFI per β); a proper calibration loop would.
* Real VAR uses straightforward log-diffs; production analysis would examine transformations and lag order selection more carefully.
* OpenMP parallelizes the backup over $z$; extending to deeper vectorization or GPU is future work.

## Reference Equations (brief)

* **Bellman:** $V(k,z)=\max_{k'\in \mathcal{K}} \{\log(z k^\alpha+(1-\delta)k-k') + \beta \mathbb{E}[V(k',z')]\}$
* **VAR(1):** $Y_t = A Y_{t-1} + c + \varepsilon_t,\quad Y_t=\begin{bmatrix}\Delta \log \text{GDP}_t \\ \Delta \log \text{CONS}_t \end{bmatrix}$

## References

* King & Rebelo (1999), *Resuscitating Real Business Cycles*, Handbook of Macroeconomics.
* Tauchen (1986), *Finite-state Markov-chain approximations…*, Economics Letters.

```
