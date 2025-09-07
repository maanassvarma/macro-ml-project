# Architecture

```
Python VFI (dsge_rbc_baseline.py)
  ├─ Tauchen discretization for z
  ├─ Value function iteration on (k,z) grid
  ├─ Optimal policy k' indices → export policy dataset
  ├─ Simulated series + IRFs → CSV artifacts
  └─ (optional) timing printouts
        │
        ├─→ NumPy MLP surrogate (dsge_policy_nn.py)
        │     - Train on (k,z) → k' pairs from optimal policy
        │     - Report train/test RMSE; error plot
        │
        ├─→ Linear VAR(1) (var_linear.py / var_real.py)
        │     - Simulated baseline and FRED data
        │     - Report coefficients and IRFs
        │
        ├─→ C++ Bellman kernel (dsge_rbc_cpp.cpp)
        │     - Same backup step, compiled with -O3
        │     - Timing via <chrono>
        │
        └─→ Fortran Bellman kernel (dsge_rbc_fortran.f90)
              - Equivalent backup loop in F90
              - Timing via system_clock
```

The compiled kernels benchmark the core dynamic programming step and demonstrate portability/performance.
