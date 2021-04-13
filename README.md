## Aula.jl: An implementation of augmented lagrangian method in Julia ![CI](https://github.com/HiroIshida/Aula.jl/workflows/CI/badge.svg)

The main algorithm of aula is implemented referring Nocedal et al. [1]. The unconstraind-subproblem is solved using Levenberg–Marquardt-like method , where the lagrangian function is approximated by a gauss-newton approximation. The solver for the subproblem is implemented referring Toussaint et. al. [2], where all the tuning parameter is set as the same in the paper.


### Reference
[1] Nocedal J, Wright S. Numerical optimization. Springer Science & Business Media; 2006 Dec 11.

[2] Toussaint M. A tutorial on Newton methods for constrained trajectory optimization and relations to SLAM, Gaussian Process smoothing, optimal control, and probabilistic inference. Geometric and numerical foundations of movements. 2017:361-92.
