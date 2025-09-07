// C++: Bellman backup sweep for RBC + timing + optional OpenMP repeats
#include <bits/stdc++.h>
#include <chrono>
#ifdef _OPENMP
  #include <omp.h>
#endif
using namespace std;

inline double u(double c){ return (c>0.0) ? log(c) : -1e-10; }

int main(){
    // ---- problem size / params
    const double alpha=0.33, beta=0.96, delta=0.08;
    const int Nk=120, Nz=7;
    const double k_min=0.5, k_max=3.0;

    // ---- grids
    vector<double> k_grid(Nk);
    for(int i=0;i<Nk;i++) k_grid[i] = k_min + (k_max-k_min)*i/(Nk-1);
    vector<double> z_grid = {0.74, 0.84, 0.95, 1.00, 1.06, 1.18, 1.30};

    // ---- dummy continuation value (single backup)
    vector<vector<double>> EV(Nk, vector<double>(Nz, 0.0));

    vector<vector<double>> V_new(Nk, vector<double>(Nz, 0.0));
    vector<vector<int>>    policy_idx(Nk, vector<int>(Nz, 0));

    // ---- repeat the kernel to get stable timings
    const int REPS = 200;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < REPS; ++r) {
        // Parallelize across z if OpenMP is enabled (each (ik,iz) is unique)
        #pragma omp parallel for schedule(static)
        for(int iz=0; iz<Nz; ++iz){
            for(int ik=0; ik<Nk; ++ik){
                double k = k_grid[ik], z = z_grid[iz];
                double y = z * pow(k, alpha);
                double best = -1e18; int arg = 0;
                for(int j=0; j<Nk; ++j){
                    double kp = k_grid[j];
                    double c  = y + (1.0 - delta)*k - kp;
                    double rhs = u(c) + beta * EV[j][iz];
                    if(rhs > best){ best = rhs; arg = j; }
                }
                V_new[ik][iz]    = best;
                policy_idx[ik][iz]= arg;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    cout << "C++ Bellman backup completed. Example: policy_idx[60][3]="
         << policy_idx[60][3] << "\n";
    cout << "C++ total time (REPS=" << REPS << "): " << dt.count() << " s\n";
    cout << "C++ per-sweep time: " << (dt.count()/REPS) << " s\n";
#ifdef _OPENMP
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif
    return 0;
}
