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

> Everything above this line documents the pipeline of the original submission (LSTM routers, `train_router.py`, `results.py`); its one retained experiment has been withdrawn from the paper and it is kept for the record only. The wall-clock study reported in the paper is the section below.

## Cost-Aware Wall-Clock Study (branch `costaware-wallclock`) — replication guide

This section documents the wall-clock experiments added for the revision
(paper Appendix "Cost-Aware Routing and Wall-Clock Evaluation", main-text
Table 2 / Figure 3). It is written so that another person (or an agent) can
reproduce every number in those tables from scratch. All of it lives next to
the original code; nothing above this section is needed for it.

### What is measured

Four linear PDEs on the periodic unit square with the paper's hierarchical
GRF forcing (real white noise filtered in Fourier space, Nyquist modes
excluded; instances drawn sequentially so that seed + count reproduce a
prefix), discretised by the same 5-point stencil as `pde.py`: Poisson,
convection--diffusion (velocity (20, 20)), anisotropic diffusion
(`-0.01 u_xx - u_yy = f`) and variable-coefficient diffusion (a fixed smooth
coefficient field of contrast 10, grid-independent, reference solutions by
sparse LU). Grids 128x128 (all experiments), 256x256 and 512x512 (isotropic
equations only at 512^2). For each classical solver (Jacobi, damped Jacobi
0.67, GS, SymGS, SOR 1.5, geometric multigrid V(2,2); line GS for anisotropic
diffusion) the ensemble is {solver, DeepONet corrector}; policies compared:

* `classical`   solver alone
* `hints<tau>`  HINTS: corrector every tau-th iteration, first at tau
                (tau in {2, 5, 10, 15, 25, 50}; 15 is the 2-D period of the HINTS paper)
* `phints<tau>` phase-shifted HINTS: first call at iteration 0 (tau in {5, 10, 15, 25, 50})
* `oneshot`     corrector once at iteration 0, then the solver only
* `decay<th>`   residual-decay rule at the router's granularity (th in {0.9, 0.95, 0.98})
* `greedy`      paper's Algorithm 1 on single iterations (cost-agnostic oracle)
* `oracle`      Algorithm 1 on cost-equalised macro-actions (cost-aware oracle)
* `router`      the learned cost-aware router (deployable; pays for its decisions)
* `base:<m>`    classical baselines without corrector, timed in the same replay
                loop: fft, mg (and mg_line for anisotropic diffusion), lu
                (variable coefficients), cg, pcg_ssor, pcg_mg, bicgstab,
                bicgstab_mg, gmres

plus solver ensembles (nested and larger sets, with every member's pairwise
router evaluated in the same session), per-operation overheads,
training-cost amortisation, five router-training seeds, an exhaustive
short-horizon check of Theorem 4.1, numerical checks of the theory
assumptions and a discretisation-error study on the confirmatory instances.

Metric: for every test instance the true relative error is recorded after
every iteration (untimed pass), then the decision sequence is replayed in a
timed pass that executes only the chosen operations (residual + norm, update;
the router re-decides live and its feature/decision costs are charged). Three
timed replays per instance in a random order over policies and baselines,
each after an untimed warm-up; a drift guard re-times a reference operation
before every instance and waits (up to 5 min) while it is more than 10%
slower than the fastest state seen so far in the session, re-times the
instances that were still too slow at the end of the cell, and `bench.py
--retime_only [--retime_all]` re-times an existing cell (all 128^2 pairwise
cells are re-timed this way at the end of the chain). We
report the median wall-clock time at which the error first drops below a
tolerance, mainly eps = h^2, paired per-instance speedups with bootstrap 95%
intervals, two-sided Wilcoxon tests on log ratios with a Holm correction
(a censored baseline enters at its time-to-cap as a lower bound, a censored router run as a ratio of 0 counted against the router, both censored as a tie), paired t-tests
on log AUC, and work units (measured per-iteration costs x executed
operations) wherever sessions must be compared.

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
| `baselines.py` | Krylov / multigrid / FFT / sparse-LU baselines as replayable methods (`make_baseline`), timed by `bench.py` inside the same loop as the policies (`base:<m>` entries of the pairwise result files); `bench_baselines.py` is the earlier separate-session driver, no longer used |
| `bench_overheads.py` | per-operation costs vs N (incl. the paper's LSTM router) and training times -> `results/overheads.json` |
| `bench_seeds.py` | five-seed router retraining trials -> `results/seeds_<eq>_<N>.json` |
| `make_usage_data.py` | untimed decision traces of all test instances (usage figures) -> `results/usage_<eq>_<N>.json` |
| `make_tables.py`, `make_figures.py` | LaTeX tables (`paper/costaware_tables.tex`, one macro per table plus summary macros used in the text; also writes `paper/manifest.json` with the sha256 of every result file used) and figures (`paper/neurips_images/ca_*.png`) |
| `discretization_error.py` | empirical discretisation error of the test instances (exact discrete solution vs. a 4x finer grid) -> `results/discretization_error.json` |
| `screen_ensembles.py` | oracle-level screening of every ensemble of up to three members (work units) -> `results/screen_<eq>_<N>.json` |
| `check_theorem.py` | exhaustive short-horizon check of Theorem 4.1 for the deployed rule (mu, alpha(O), bound, clipped optima, violations) -> `results/theorem_<eq>_<N>.json` |
| `check_assumptions.py` | numerical checks of the theory assumptions (Lipschitz constants in the Euclidean and energy norms, spectral radii, invertibility, zero preservation, commutators, alpha(O), Thm 5.1 bounds) -> `results/assumptions_<eq>_<N>.json` |
| `run_final.sh`, `run_varcoeff.sh`, `run_granularity.sh`, `run_all.sh` | the exact sequence of commands that produces the reported results (revision 2); the older `run_*.sh` scripts produced the development runs |

### Step-by-step replication

```
# 0. build the compiled stencil kernels (all sweeps incl. line GS, residual + norm, multigrid transfers) and check them
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
#    (the variable-coefficient correctors are trained by run_varcoeff.sh)
#    -> checkpoints/deeponet_<eq>_<N>_best.pth (validation relative error ~2e-6 is expected)

# 2. the confirmatory study reported in the paper (~1-2 days on an M4 Pro, sequential, idle machine):
#    run_final.sh (128^2 pairwise + same-session baselines, 128^2 ensembles, 256^2, 512^2, usage traces,
#    seed trials, overheads, assumption / screening / theorem checks, discretisation study), then
#    run_varcoeff.sh (variable-coefficient diffusion), then run_granularity.sh (unit = corrector / 4).
#    Solver lists per equation are read from config/solvers_<eq> at the start of each stage; every
#    script is fail-fast (a failing command writes logs/*.failed and stops the chain).
mkdir -p logs && PY=$(which python) nohup ./run_all.sh > logs/run_all.out 2>&1 &
#    Development runs (numpy kernels, seed 72, earlier sampler) on which every configuration was chosen
#    are archived in results_dev/; the first confirmatory run (before the protocol revision) in results_conf1/.

# 3. tables, figures, paper
python make_tables.py   # -> paper/costaware_tables.tex (+ paper/manifest.json with the sha256 of every result file used)
python make_figures.py  # -> paper/neurips_images/ca_*.png
paper/build.sh          # -> paper/neurips_2026.pdf (plain pdflatex/bibtex in a scratch dir)
```

Individual pieces can be run by hand, e.g. one pairing:

```
S=jacobi,jacobi_0.67,gs,ssor,sor_1.5,mg
python bench.py --equation Poisson --N 128 --solvers $S --measure_only --remeasure_costs   # shared per-iteration costs of all operations -> checkpoints/costs_Poisson_128.json
python bench.py --equation Poisson --N 128 --solvers gs --train_only --retrain_router    # router -> checkpoints/router_Poisson_128_gs.pth
python bench.py --equation Poisson --N 128 --solvers gs --n_test 64 --seed 73 --timed_reps 3   # benchmark (revised protocol) -> results/Poisson_128_gs.json
#   (the default seed is the development seed 72 with one replay; measure the costs with the full solver
#    list first, as run_final.sh does, so that every pairing shares one calibration; delete the cost cache
#    and retrain the routers when moving to another machine)
```

Useful options of `bench.py`: `--policies` (comma list), `--ensemble`
(all `--solvers` in one ensemble), `--max_ops` (iteration cap; solver-only
runs that hit it are reported as lower bounds), `--retrain_router`,
`--dagger_rounds`, `--router_inst`, `--router_hidden`, `--router_epochs`,
`--rate` (per-iteration form of the cost-aware rule, not used in the paper).

### Conventions and pitfalls

* The unit of cost is one corrector call; operation j is applied
  `m_j = max(1, round(u / c_j))` times per decision and these macro-actions
  are compared by their plain error (Algorithm 1; exponent 1 for every action
  not dearer than the unit); only an operation dearer than the unit (a
  multigrid cycle at 512^2, or the corrector in the granularity ablation) is
  applied once and compared per unit of cost (`Env.macro_score`). Costs are
  measured once per (equation, grid) in the exact form the replay charges
  (update + residual + norm; median over blocks) and cached in
  `checkpoints/costs_<eq>_<N>.json`; delete the cache to re-measure on a new
  machine (all routers must then be retrained).
* Confirmatory test instances are seed 73 (same for every policy and
  baseline); every configuration choice was made on the development set,
  seed 72. Router training instances use seed 555 (+ per-seed offsets in
  `bench_seeds.py`), corrector training seed 1234. Two baselines are
  deliberately optimistic and labelled as such: the "best fixed schedule" and
  the "best residual-decay rule", both selected per cell on the test set.
* Every result file records its provenance (git commit and dirty flag,
  compiled-kernel and router checkpoint hashes, package versions, per-instance
  drift ratios and load averages); `paper/manifest.json` lists the sha256 of
  every file the tables were built from.
* Live times include the residual evaluation of the stopping test in every
  iteration for every method. `t_wu` in the result files is a timer-free
  cross-check (measured per-iteration costs x executed operations).
* The repository directory on the original machine sits under an
  iCloud-synced folder; `paper/build.sh` therefore builds in `/private/tmp`
  and copies the PDF back. On other machines a plain `latexmk -pdf` in
  `paper/` also works.
* `logs/`, `checkpoints/*.pth` and the cost caches are not tracked. The result
  files of the revision-2 confirmatory run are committed to `results/` when the
  chain finishes (until then only the table file `paper/costaware_tables.tex`
  and `paper/manifest.json` with the sha256 of every file it was built from are
  tracked); the development runs are in `results_dev/`, the first confirmatory
  run in `results_conf1/` (untracked).
