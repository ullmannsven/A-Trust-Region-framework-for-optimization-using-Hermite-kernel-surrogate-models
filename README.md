# A Trust-Region Framework for Optimization using Hermite Kernel Surrogate Models

Reference implementation accompanying the paper

> **A trust-region framework for optimization using Hermite kernel surrogate models**
> S. Ullmann, T. Ehring, R. Herkert, B. Haasdonk

This repository contains the Python code that reproduces all numerical experiments reported in the paper. The core contribution is the **HKTR algorithm** (Hermite-Kernel Trust-Region, Algorithm 2 in the paper): a derivative-aware surrogate-based trust-region method for parameter optimization problems where each evaluation of the objective requires solving a (possibly expensive) full-order model (FOM), typically a PDE.

## What the method does

For a parameter-to-output map `J : μ ↦ J(μ)` whose evaluation involves solving a PDE, classical optimizers (BFGS, trust-constr, ROL) call the FOM and its adjoint many times. HKTR replaces these calls with a *Hermite kernel surrogate* that interpolates **both function values and gradients** of `J` at a small set of training parameters, and embeds the surrogate in a trust-region loop with a rigorous a-posteriori error estimator based on the kernel power function and the RKHS norm. The surrogate is updated on the fly: new FOM evaluations are added when needed, and points are removed when the Gram matrix becomes ill-conditioned or when they fall outside the current trust region. The numerical examples in the paper (and in this repo) show that this leads to reductions in the number of FOM evaluations needed to reach a given optimization tolerance.

## Repository layout

```
.
├── functions/                 # Core algorithm code
│   ├── kernel.py              # Kernel classes + Hermite Gram matrices
│   ├── kernel_width_hermite_TR.py   # Main HKTR algorithm (Algorithm 2)
│   ├── model.py               # The test problems
│   ├── results_analysis.py    # runs HKTR over several kernel shape parameters and initial values
│   ├── scipy_algos.py         # Baselines: L-BFGS-B and trust-constr
│   └── rol_setup.py           
├── pyMORAuxData/              # Problem definitions for the pyMOR-based examples
│   ├── problems.py            # 2D linear elliptic, 12D building-floor problem
│   ├── twelve_dim_*.py        # 12D discretizer/model/tools (from Keil et al.)
│   └── EXC_data/              # Bitmap data describing the 12D building geometry
├── examples/                  # Entry points: one script per experiment
│   ├── run_1D_hktr.py         # Reproduces the 1D row of Table 1
│   ├── run_2D_hktr.py         # Reproduces the 2D row of Table 3
│   ├── run_12D_hktr.py        # Reproduces the 12D row of Table 5
│   ├── run_4d_nonlinear_hktr.py   # Semilinear PDE identification (9 weights)
│   ├── run_4d_rol.py          # Same problem, solved with pyROL
│   └── run_*_scipy_*.py       # SciPy baselines for Tables 2, 4, 6
└── README.md                  
```

## The kernels (`functions/kernel.py`)

Five radial kernels are implemented; each provides the value `φ(r)`, the rescaled first derivative `φ'(r)/r`, the rescaled second derivative `(φ'/r)'/r`, and a NumPy assembly of the Hermite Gram matrix. The Gauss, QuadMatern, and InvMulti classes additionally provide PyTorch versions used when the kernel width is treated as an optimization variable (`gamma_adaptive=True`). Note that in the current version of the paper, results regarding an adaptive shape parameter are not reported. 

| Class | Kernel | Used in |
|---|---|---|
| `Gauss` | `exp(-γ r²)` | 1D example |
| `QuadMatern` | `exp(-γ r) (3 + 3γr + γ²r²)` | 2D example |
| `QuadWendland` | Wendland-type compactly supported kernel | 12D and 9D nonlinear example |
| `InvMulti` | Inverse multiquadric | available |
| `LinMatern` | Linear Matern | available |

The Hermite Gram matrix is a block matrix which makes the surrogate interpolate both `J` and `∇J` at every training point. Implementation is based on **VKOGA** (https://github.com/GabrieleSantin/VKOGA), with the Hermite extension specific to this work, following https://link.springer.com/article/10.1007/s10444-024-10128-5. 

## The HKTR algorithm (`functions/kernel_width_hermite_TR.py`)

The main entry point is `tr_Kernel(model, kernel, TR_parameters)`. One outer iteration does the following:

1. Build the Hermite kernel surrogate `s_k` by interpolating `(J, ∇J)` at the current set of training points.
2. **Subproblem (`solve_subproblem_scipyBFGS`):** minimize `s_k` using SciPy's L-BFGS-B with a custom callback that terminates as soon as the iterate reaches the trust-region boundary (measured via the kernel power function × RKHS norm vs. `β₂ · radius`), and a penalty term enforcing the trust-region constraint inside the subproblem objective.
3. Use the kernel a-posteriori estimator `‖J - s_k‖ ≤ P_X(μ) · ‖J‖_RKHS` to decide whether to **accept** the candidate, **reject** it (and shrink the radius by `β₁`), or accept conditionally after a FOM check.
4. Update the training set: append the new point, remove points farther than `max_amount_interpolation_points` from the iterate (`remove_far_away_points`), and remove near-duplicates that would push the Gram matrix above `cond_threshold` (`remove_similar_points`).
5. Optionally enlarge the radius (factor `1/β₁`) when the actual-vs-predicted reduction ratio exceeds `ρ`.

If `gamma_adaptive=True`, the kernel width γ is appended as an extra coordinate to the parameter, and its gradient is computed via PyTorch autograd (`compute_gradientGamma`). In the experiments shipped here `gamma_adaptive=False` is used and γ is swept manually via `gamma_list`.

The RKHS norm needed for the error estimator is either supplied analytically by the model (`compute_RKHS_norm`, available only for `Gaussian1D`) or estimated from 10 random samples (`computeDataForRKHSNorm`).

### Key TR parameters (passed via the `TR_parameters` dict)

| Parameter | Meaning |
|---|---|
| `radius` | initial trust-region radius |
| `sub_tolerance` | tolerance for the inner BFGS subproblem |
| `max_iterations`, `max_iterations_subproblem` | outer / inner iteration caps |
| `FOC_tolerance`, `J_tolerance` | first-order criticality and objective stopping tolerances |
| `beta_1`, `beta_2` | radius shrink factor and TR boundary safety factor |
| `rho` | actual/predicted reduction threshold for radius enlargement |
| `max_amount_interpolation_points` | size cap on the kernel training set |
| `cond_threshold` | maximum tolerated condition number of the Gram matrix |
| `gamma_adaptive` | whether to optimize γ jointly with μ |

## The four test problems (`functions/model.py`)

| Class | dim | Type | Source / FOM |
|---|---|---|---|
| `Gaussian1D` | 1 | Closed-form, two Gaussians | analytic; analytic RKHS norm available |
| `twoDStuff` | 2 | 2D linear elliptic PDE-constrained problem | pyMOR, `discretize_stationary_cg` |
| `buildingFloor` | 12 | Stationary heat distribution on a building floor with parametric walls / doors / heaters | pyMOR + bitmap geometry from `pyMORAuxData/EXC_data/`; reused from [Keil et al.](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt) |
| `NonlinearModel` | 9 | Semilinear PDE parameter identification: recover the 9 weights of a Gaussian basis for `σ` in `-Δu + σ(w) u³ = f` from a reference state | FEniCSx (`dolfinx`), Newton via PETSc SNES, adjoint-based gradient |

All models expose the same `getFuncAndGradient(μ)` interface returning `(J(μ), ∇J(μ))` and a counter `fomCounter` that is incremented on every full-order solve — this is the metric the experiments report.

## Reproducing the paper

The following table maps scripts to paper artefacts (per the original `readme.md`).

| Script | Reproduces |
|---|---|
| `examples/run_1D_hktr.py`  | 1D row of Table 1 |
| `examples/run_2D_hktr.py`  | 2D row of Table 3 |
| `examples/run_12D_hktr.py` | 12D row of Table 5 |
| `examples/run_1D_scipy_bfgs.py` / `..._trust_constr.py` | L-BFGS-B / trust-constr rows of Table 2 |
| `examples/run_2D_scipy_*.py` | corresponding rows of Table 4 |
| `examples/run_12D_scipy_bfgs.py` | L-BFGS-B row of Table 6 |
| `examples/run_4d_nonlinear_hktr.py` | HKTR results on the 9-parameter semilinear identification |
| `examples/run_4d_rol.py` | pyROL Lin-More baseline on the same problem |
| `examples/run_4d_scipy_*.py` | SciPy baselines on the same problem |

Each `run_*_hktr.py` script declares
- a list `gamma_list` of kernel widths to sweep,
- `amount_of_iters = 5` random starting points (fixed seeds 0..4),
- a `TR_parameters` dictionary,

then calls `optimize_all` (which runs HKTR for every (γ, seed) pair) and prints a results table averaged over starting points.

Example: `examples/run_2D_hktr.py`

```python
import functions.model as models
import functions.results_analysis as result_analysis

gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
amount_of_iters = 5
model = models.twoDStuff()

TR_parameters = {
    'radius': 1, 'sub_tolerance': 1e-4,
    'max_iterations': 100, 'max_iterations_subproblem': 30,
    'FOC_tolerance': 1e-4, 'J_tolerance': 1e-12,
    'beta_1': 0.5, 'beta_2': 0.95, 'rho': 0.9,
    'max_amount_interpolation_points': 10,
    'cond_threshold': 1e20, 'gamma_adaptive': False,
}

optim_data = result_analysis.optimize_all(model, gamma_list, TR_parameters, amount_of_iters)
result_analysis.report_kernel_TR(optim_data, gamma_list, amount_of_iters)
```

The output is a pandas DataFrame with columns `gamma`, `avg. FOM evals.`, `avg. FOC condition`, `avg. error in J`.

## Dependencies

The code combines a numerical-analysis stack with two PDE frameworks:

- **Core numerics:** NumPy, SciPy, pandas, matplotlib, PyTorch (used for autograd over shape parameter, not part of the current version of the paper).
- **pyMOR** (https://github.com/pymor/pymor) — drives the 2D and 12D PDE-constrained problems and provides the parameter-space machinery.
- **FEniCSx / DOLFINx** (`dolfinx`, `ufl`, `mpi4py`, `petsc4py`) — used by the 9-parameter semilinear identification problem (`NonlinearModel`).
- **pyROL** (Trilinos ROL Python bindings); only required for `examples/run_4d_rol.py`.

The 12D files in `pyMORAuxData/twelve_dim_*.py` are reused from [Tim Keil's `Proj-Newton-NCD-corrected-TR-RB-for-pde-opt`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt) and contain more functionality than is exercised by the experiments here. The kernel code in `functions/kernel.py` is based on **VKOGA** (https://github.com/GabrieleSantin/VKOGA) extended with the Hermite Gram matrix construction.

## Running an experiment

From the repository root:

```bash
python -m examples.1D.run_1D_hktr
```

The 1D example is light and finishes in seconds. The 2D and 4D examples run in minutes on a laptop. The 12D building-floor example is heavy — each FOM evaluation involves solving a parametric stationary diffusion problem on a fine mesh, and one full sweep over `gamma_list × amount_of_iters` can take a while.

Console output during a run includes per-iteration trust-region radius adjustments, the candidate parameter, the surrogate-vs-FOM reduction, and removal events when points are pruned from the kernel training set.