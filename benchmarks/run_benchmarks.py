"""
Benchmark core kernels:
- Python Bellman backup (inline)
- C++ bellman_demo (hpc/)
- Fortran bellman_fortran (hpc/, optional)

Writes benchmarks/benchmarks.json and prints it.
"""
import os, time, json, subprocess, numpy as np

def python_backup_time(Nk=120, Nz=7, reps=3):
    alpha=0.33; delta=0.08; beta=0.96
    k_grid = np.linspace(0.5, 3.0, Nk)
    z_grid = np.array([0.74, 0.84, 0.95, 1.00, 1.06, 1.18, 1.30])
    EV = np.zeros((Nk, Nz))
    def backup():
        V_new = np.empty((Nk,Nz))
        pol   = np.empty((Nk,Nz), dtype=int)
        for iz in range(Nz):
            z = z_grid[iz]
            for ik in range(Nk):
                k = k_grid[ik]
                y = z * (k**alpha)
                best = -1e18; arg=0
                for j in range(Nk):
                    kp = k_grid[j]
                    c  = y + (1.0 - delta)*k - kp
                    rhs = (np.log(c) if c>0 else -1e10) + beta*EV[j,iz]
                    if rhs>best: best=rhs; arg=j
                V_new[ik,iz]=best; pol[ik,iz]=arg
        return V_new, pol
    ts=[]
    for _ in range(reps):
        t0=time.time(); backup(); t1=time.time(); ts.append(t1-t0)
    return float(np.median(ts))

def run(path):
    if not os.path.exists(path):
        return None, "missing"
    try:
        out = subprocess.check_output([path], stderr=subprocess.STDOUT, timeout=120).decode()
        return out, "ok"
    except Exception as e:
        return str(e), "error"

def parse_per_sweep(output, key="per-sweep time"):
    # expects a line like: "C++ per-sweep time: 0.000633281 s"
    for line in (output or "").splitlines():
        if key in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except Exception:
                pass
    return None

def main():
    res = {}
    res["python_backup_median_s"] = python_backup_time()

    out_cpp, st_cpp = run("./hpc/bellman_demo")
    out_ftn, st_ftn = run("./hpc/bellman_fortran")

    res["cpp_status"]    = st_cpp
    res["fortran_status"]= st_ftn
    res["cpp_output"]    = out_cpp
    res["fortran_output"]= out_ftn
    res["cpp_per_sweep_s"]     = parse_per_sweep(out_cpp)
    res["fortran_per_sweep_s"] = parse_per_sweep(out_ftn, key="per-sweep time")

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/benchmarks.json","w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
