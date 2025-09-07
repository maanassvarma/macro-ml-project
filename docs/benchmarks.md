# Benchmarks

Record machine & versions here:
- CPU / RAM:
- Python:
- NumPy:
- g++ / gfortran:

## Results (example format)
| Kernel                 | Size (Nk,Nz) | Time (s) | Notes |
|------------------------|--------------|----------|-------|
| Python backup (median) | 120,7        | 0.XX     | bench via benchmarks/run_benchmarks.py |
| C++ bellman_demo       | 120,7        | 0.XX     | parsed from program output |
| Fortran backup         | 120,7        | 0.XX     | parsed from program output |

See `benchmarks/benchmarks.json` for raw outputs.

## Measured on this machine
| Python backup (median) | 120,7 | 0.066224 | inline kernel |
| C++ per-sweep          | 120,7 | 0.000441888 | bellman_demo |
| Fortran per-sweep      | 120,7 | 0.000398865 | bellman_fortran |
