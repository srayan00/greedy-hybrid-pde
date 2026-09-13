# A Greedy PDE Router for Blending Neural Operators and Classical Methods
  
## Install Dependencies
Run the following command to install all required dependencies
`conda env create -f environment.yml`

## Training DeepONet
See Table 3 in Appendix D for the exact hyperparameters used in our DeepONet and port them over to args/deeponet_args.json. Run the following commands:
`conda activate greedy`
`python train_ml_solver.py --model_name ML_MODEL_NAME --equation  [Poisson/ConvDiff]`

For example, 
`python train_ml_solver.py --model_name ml_example --equation  Poisson`


## Training a Greedy Router

Run the following command:
`python train_router.py --ml_model_name ML_MODEL_NAME --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
where `LIST_OF_SOLVERS` is a comma-separated list of solvers in the solver ensemble.

For example, 
`python train_router.py --ml_model_name ml_example --model_name example --equation Poisson --numerical_solvers jacobi_0.8,gs,sor_1.5,ssor`
where `jacobi_0.8` is a Weighted Jacobi solver with a relaxation parameter $\omega = 0.8$ and `gs` denotes Gauss-Seidel method, `sor_1.5` denotes a successive over-relaxation solver ($\omega = 1.5$), and `ssor` denotes a symmetric successive over-relaxation solver ($\omega = 1.0$)

## Running Experiments

### Comparing Greedy with HINTS experiment

Train routers for `equation` = `Poisson` and `ConvDiff` for the following list of solver ensembles `[jacobi, gs, ssor, jacobi_0.67, sor_1.5]` . There should be a total of $10$ routers ($2 \times 5$) 

After all these models are trained, run the command:
`python results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers [jacobi/gs/ssor/jacobi_0.67/sor_1.5]`
for all 10 combinations. All the results (plots and tables) can be in the results folder

### Size of solver ensembles
Train routers for `equation` = `Poisson` and `Helmholtz` for the following list of solver ensembles:
* `jacobi,gs`
* `jacobi,gs,ssor`
* `jacobi,gs,ssor,jacobi_0.67`
* `jacobi,gs,ssor,jacobi_0.67,sor_1.5`


There should be a total of $8$ routers ($2 \times 4$) 

After all these models are trained, run the command:
`python multiple_solver_results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
for all $8$ combinations. All the results (plots and tables) can be found in the results folder

## Cost-Aware Wall-Clock Study (branch `costaware-wallclock`) — replication guide

This section documents the wall-clock experiments added for the revision
(paper Appendix "Cost-Aware Routing and Wall-Clock Evaluation", main-text
Table 2 / Figure 3). It is written so that another person (or an agent) can
reproduce every number in those tables from scratch. All of it lives next to
the original code; nothing above this section is needed for it.

### What is measured

Three linear PDEs on the periodic unit square with the paper's hierarchical
GRF forcing, discretised by the same 5-point stencil as `pde.py`:
Poisson, convection--diffusion (velocity (20, 20)) and anisotropic diffusion
(`-0.01 u_xx - u_yy = f`). Grids 128x128 (all experiments), 256x256 and
512x512 (scaling). For each classical solver (Jacobi, damped Jacobi 0.67, GS,
SymGS, SOR 1.5, and geometric multigrid V(2,2)) the ensemble is
{solver, DeepONet corrector}; policies compared:

* `classical`  solver alone
* `hints<tau>` HINTS: corrector every tau-th iteration (tau in {5, 10, 25, 50})
* `greedy`     paper's Algorithm 1 on single iterations (cost-agnostic oracle)
* `oracle`     Algorithm 1 on cost-equalised macro-actions (cost-aware oracle)
* `router`     the learned cost-aware router (deployable; pays for its decisions)

plus strong classical baselines (FFT direct solve, multigrid alone, CG,
PCG-SymGS, PCG-multigrid, BiCGSTAB, BiCGSTAB-multigrid, GMRES(20)), solver
ensembles, per-operation overheads, training-cost amortisation, and paired
significance tests with five router-training seeds.

Metric: for every test instance the true relative error is recorded after
every iteration (untimed pass), then the decision sequence is replayed in a
timed pass that executes only the chosen operations (the router re-decides
live and its feature/decision costs are charged). We report the median
wall-clock time at which the error first drops below a tolerance, mainly
eps = h^2 (truncation level), paired per-instance speedups, one-sided
Wilcoxon / paired t-tests on log times, and the paper's iteration-based AUC.

### Environment

* macOS (Apple M4 Pro, single-thread timing) with conda; the exact Python
  environment used is `ansatz` (Python 3.11, torch 2.13, numpy 2.4,
  scipy 1.17, matplotlib 3.11, pypdfium2 for PDF checks). Any Python >= 3.10
  with numpy/scipy/torch/matplotlib works; MPS/CUDA is not required (the
  corrector is fit by least squares on the CPU).
* Timing runs must be executed one at a time on an otherwise idle machine
  with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`
  (the run scripts set this). Absolute times depend on the machine; paired
  speedups and iteration counts are the transferable quantities.
* All commands below are run from the repository root with the environment's
  python on the PATH (replace `python` by the full interpreter path if needed).

### Files

| file | role |
|---|---|
| `fast_pde.py` | O(N^2) stencil operators (Poisson / ConvDiff / AnisoDiff), FFT direct solver (ground truth), Jacobi / GS / SOR / SSOR / multigrid steps, GRF sampler; validated against `pde.py` by `validate_fast_pde.py` |
| `corrector.py` | the DeepONet corrector: band-limited FFT restriction to a sensor grid, fixed orthonormal Fourier trunk, linear branch fit by least squares on exact residual/error pairs, scale-equivariant application (`DeepONetCorrector`) |
| `hybrid.py` | cost-equalised macro-actions (`Env`), router features (`FeatureState`), untimed rollouts, timed replay, work-unit accounting |
| `router.py` | numpy MLP router, batched oracle-labelled data collection, paper's cost-weighted surrogate loss, DAgger training (`fit_router`) |
| `bench.py` | pairwise / ensemble benchmark driver (costs -> routers -> untimed + timed passes -> `results/<eq>_<N>_<solver>.json`) |
| `baselines.py`, `bench_baselines.py` | Krylov / multigrid / FFT baselines -> `results/baselines_<eq>_<N>.json` |
| `bench_overheads.py` | per-operation costs vs N (incl. the paper's LSTM router) and training times -> `results/overheads.json` |
| `bench_seeds.py` | five-seed router retraining trials -> `results/seeds_<eq>_<N>.json` |
| `make_usage_data.py` | untimed decision traces of all test instances (usage figures) -> `results/usage_<eq>_<N>.json` |
| `make_tables.py`, `make_figures.py` | LaTeX tables (`paper/costaware_tables.tex`, one macro per table plus summary macros used in the text; also writes `paper/manifest.json` with the sha256 of every result file used) and figures (`paper/neurips_images/ca_*.png`) |
| `discretization_error.py` | empirical discretisation error of the test instances (exact discrete solution vs. a 4x finer grid) -> `results/discretization_error.json` |
| `screen_ensembles.py` | oracle-level screening of every ensemble of up to three members (work units) -> `results/screen_<eq>_<N>.json` |
| `check_assumptions.py` | numerical verification of the theory assumptions (Lipschitz constants in the Euclidean and energy norms, spectral radii, invertibility, zero preservation, commutators, alpha(O), Thm 5.1 bounds) -> `results/assumptions_<eq>_<N>.json` |
| `run_correctors.sh`, `run_correctors_large.sh`, `run_all.sh`, `run_phase6.sh`, `run_aniso.sh`, `run_phase7.sh`, `run_usage_large.sh`, `run_screen.sh`, `run_assumptions.sh`, `run_ens_nested.sh`, `run_retime.sh`, `run_paths.sh`, `run_paths_h2.sh` | the exact sequence of commands that produced the reported results |

### Step-by-step replication

```
# 0. build the compiled stencil kernels (all classical sweeps, residual, multigrid transfers) and check them
cc -O3 -shared -fPIC -o libstencil.so stencil.c     # STENCIL_NUMPY=1 forces the numpy/scipy fallback
python validate_fast_pde.py                        # asserts against the dense reference at N=31 (rel. tol. 1e-6)

# 1. correctors: one per (equation, grid). Sensor grid = N/coarsen per axis;
#    64x64 for the isotropic equations, full x-resolution for AnisoDiff.
python corrector.py --equation Poisson   --N 128 --coarsen 2
python corrector.py --equation ConvDiff  --N 128 --coarsen 2
python corrector.py --equation Poisson   --N 256 --coarsen 4 --n_train 32000 --n_val 1000
python corrector.py --equation ConvDiff  --N 256 --coarsen 4 --n_train 32000 --n_val 1000
python corrector.py --equation Poisson   --N 512 --coarsen 8 --n_train 32000 --n_val 1000
python corrector.py --equation ConvDiff  --N 512 --coarsen 8 --n_train 32000 --n_val 1000
python corrector.py --equation AnisoDiff --N 128 --coarsen_x 1 --coarsen_y 4
python corrector.py --equation AnisoDiff --N 256 --coarsen_x 1 --coarsen_y 4 --n_train 48000 --n_val 1000
#    -> checkpoints/deeponet_<eq>_<N>_best.pth (validation relative error ~2e-6 is expected)

# 2. everything at 128^2 for Poisson/ConvDiff (pairwise incl. multigrid pairing,
#    ensembles, usage traces), then 256^2, 512^2, overheads and seed trials:
./run_all.sh            # ~10 h sequential on an M4 Pro
# 3. strong classical baselines at 128^2 (256^2 / 512^2 are inside run_all.sh)
python bench_baselines.py --equation Poisson  --N 128 --n_test 64
python bench_baselines.py --equation ConvDiff --N 128 --n_test 64
# 4. larger-capacity ensemble routers (hyperparameter variant; optional)
./run_phase6.sh
# 5. anisotropic diffusion (correctors, pairwise, ensembles, usage, 256^2, seeds; no multigrid/Krylov
#    baselines: point-smoothed multigrid is not competitive on this equation)
./run_aniso.sh          # ~8 h
# 5b. (optional) retrain the one 512^2 router that misses h^2 with more oracle rollouts
./run_phase7.sh
# 6. usage traces on the larger grids, ensemble screening, nested ensembles
#    ({J} < {J, dJ} < {J, dJ, GS} with same-session pairwise baselines), theory assumptions
./run_usage_large.sh     # ~1 h
./run_screen.sh          # ~1.5 h
./run_ens_nested.sh      # ~6 h (training + benchmark; live times may be inflated if other jobs run)
./run_retime.sh          # ~3.5 h: re-time every nested cell on an idle machine (no retraining)
./run_assumptions.sh     # ~1 h
./run_final.sh           # ~20 h: the confirmatory study reported in the paper (compiled kernels, test seed 73,
                         #  frozen recipes, full fixed-schedule family, 3 timed replays); results_dev/ holds the
                         #  development runs (numpy kernels, seed 72) on which every configuration was chosen
# 7. tables, figures, paper
python make_tables.py   # -> paper/costaware_tables.tex
python make_figures.py  # -> paper/neurips_images/ca_*.png
paper/build.sh          # -> paper/neurips_2026.pdf (plain pdflatex/bibtex in a scratch dir)
```

Individual pieces can be run by hand, e.g. one pairing:

```
python bench.py --equation Poisson --N 128 --solvers gs --measure_only       # per-iteration costs -> checkpoints/costs_Poisson_128.json
python bench.py --equation Poisson --N 128 --solvers gs --train_only         # router -> checkpoints/router_Poisson_128_gs.pth
python bench.py --equation Poisson --N 128 --solvers gs --n_test 64          # benchmark -> results/Poisson_128_gs.json
```

Useful options of `bench.py`: `--policies` (comma list), `--ensemble`
(all `--solvers` in one ensemble), `--max_ops` (iteration cap; solver-only
runs that hit it are reported as lower bounds), `--retrain_router`,
`--dagger_rounds`, `--router_inst`, `--router_hidden`, `--router_epochs`,
`--rate` (per-iteration form of the cost-aware rule, not used in the paper).

### Conventions and pitfalls

* The unit of cost is one corrector call; operation j is applied
  `m_j = max(1, round(u / c_j))` times per decision; an operation dearer than
  the unit (a multigrid cycle) is applied once and compared per unit of cost
  (`Env.macro_score`). Costs are measured once per (equation, grid) with a
  min-over-blocks estimator and cached in `checkpoints/costs_<eq>_<N>.json`;
  delete the cache to re-measure on a new machine.
* Test instances are seed 72 (same for every policy and baseline), router
  training instances seed 555 (+ per-seed offsets in `bench_seeds.py`),
  corrector training seed 1234. Nothing is tuned on the test set except the
  "best fixed tau" HINTS baseline, which is deliberately optimistic.
* Live times include the residual evaluation of the stopping test in every
  iteration for every method. `t_wu` in the result files is a timer-free
  cross-check (measured per-iteration costs x executed operations).
* The repository directory on the original machine sits under an
  iCloud-synced folder; `paper/build.sh` therefore builds in `/private/tmp`
  and copies the PDF back. On other machines a plain `latexmk -pdf` in
  `paper/` also works.
* `logs/` and `checkpoints/*.pth` are not tracked; `results/*.json` (all
  reported numbers) are.
