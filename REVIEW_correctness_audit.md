# Reviewer report (correctness audit): *A Greedy PDE Router for Blending Neural Operators and Classical Methods*

Audit date: 2026-09-13. Repository state: branch `costaware-wallclock`, HEAD `d654a2e`, working tree dirty (see Section 1). This report is written in the voice of a conference reviewer whose primary charge is the correctness of the theory and of the experiments, with broader concerns listed separately and ranked lower.

A separate file `REVIEW.md` (untracked, written by another session at 13:30 today) already contains a review. I did not modify it. Section 8 states where this audit confirms it, extends it, and in one place contradicts it.

---

## 0. Recommendation and one-paragraph summary

**Recommendation: major revision (reject in the present form).** Suggested score 3/10, confidence high on the correctness items below, since each was checked against the source, the result files, or a direct numerical test.

The empirical observation at the centre of the revision is real and reproducible from the supplied artefacts: when a corrector that solves the smooth half of the spectrum essentially exactly is available, calling it once at the start and then smoothing reaches truncation-level error far sooner than a HINTS schedule that waits τ−1 sweeps before its first call, and a tiny scalar-feature router imitates the cost-aware oracle almost perfectly. The repository is unusually complete and the appendix is candid about several limitations.

However, (i) the main approximation theorem is not established under its stated hypotheses, because the postfix-monotonicity claim in Proposition 4.2 is false and its proof uses an invalid step; a second step of the same proof is also invalid, although that part of the result survives with a different constant. (ii) The guarantee, even where it holds, is weaker than the trivial contraction bound in every regime the paper studies. (iii) The deployed "DeepONet" is a linear least-squares fit of the exact band-limited inverse of the discrete operator, so the results do not speak to neural-operator routing, and the natural classical baselines that this reveals (a band-limited FFT correction, or the full FFT solve at 0.32 ms) are faster than the router. (iv) The wall-clock margins over HINTS and over multigrid are largely an artefact of comparing BLAS-speed corrector calls with numpy/scipy-speed stencil sweeps; a plain C sweep compiled on the same machine is 4–5× faster than the sweeps that were timed, which removes the multigrid claim and most of the best-τ margin. (v) The HINTS baseline as implemented differs from the paper's own formula for it, and the difference (whether the first corrector call is at t=0 or at t=τ) is exactly what drives the headline speedups. (vi) Several quantitative statements in the text are contradicted by the repository (SymGS equivalence to 1e-12, cost equalisation to within 8%, invertibility of GS, standard errors that are standard deviations), and the PDF is out of sync with the results directory, with a benchmark still running at audit time.

---

## 1. What was inspected and how

Read in full: `paper/neurips_2026.tex`, `paper/costaware_appendix.tex`, `paper/costaware_tables.tex`, `README.md`, `hybrid.py`, `router.py`, `bench.py`, `fast_pde.py`, `corrector.py`, `baselines.py`, `bench_baselines.py`, `check_assumptions.py`, `make_tables.py`, `bench_seeds.py`, `validate_fast_pde.py`, the cached cost files `checkpoints/costs_*.json`, `results/overheads.json`, `results/assumptions_*.json`, `results/baselines_*.json`, and a spot-check of `results/Poisson_128_jacobi.json` against the tables.

Executed: `validate_fast_pde.py` in the `ansatz` environment; `make_tables.py` regenerated to a temporary path and diffed against the tracked table file; a small C stencil library (`/tmp/stencil.c`, `-O3`, single thread) timed against the repository's numpy/scipy sweeps and the corrector on the same machine; three short numerical checks of proof steps.

Repository state at audit time, relevant to reproducibility:

- `bench.py --equation ConvDiff --N 512 --solvers jacobi_0.67,ssor` (PID 54843) had been running since 12:54 and was at instance 8/16 for SymGS. Results are therefore still changing.
- Three new result files (`Poisson_512_jacobi_0.67.json`, `Poisson_512_ssor.json`, `ConvDiff_512_jacobi_0.67.json`) are present but untracked, and the PDF (built 12:00) predates them.
- During the audit another process edited 27+ working-tree files. The edits seen by the end of the audit: the dense SSOR line in `numerical_solver.py:195` corrected to a product; `validate_fast_pde.py` given relative tolerances and assertions; `bench.py` given a `provenance` record (commit, dirty flag, package versions, checkpoint hash, run id); `hybrid.py` given a `oneshot` policy (corrector at `t = 0`, then the solver), which is the ablation requested in E2 and Section 4; run scripts and `.gitignore` touched. The `work_units` function was not changed, which is correct (see E11 and Section 8). Everything below refers to HEAD `d654a2e` unless stated otherwise.

---

## 2. Theory: correctness findings

### T1 (major). Postfix monotonicity in Proposition 4.2 is false as stated, and its proof is invalid

Theorem 4.1 needs postfix monotonicity, `h(S ⊕ N) ≤ h(N)` for all `S, N`. Proposition 4.2 claims this follows from Lipschitz constants `ρ_j < 1`, zero preservation and invertibility. The proof (Appendix B.3, "Postfix monotonicity") inserts `G_N^{-1} G_N` and bounds the Lipschitz constant of `G_N^{-1}` by `∏ ρ_{N_t}^{-1}`. That is not a property of contractions: `‖G^{-1}‖ = 1/σ_min(G)`, not `1/‖G‖`. Numerically, for `G_N = diag(0.1, 0.9)` one has `‖G_N‖ = 0.9` but `‖G_N^{-1}‖ = 10`, not `1.11`.

The claim itself is false for non-commuting maps. With `G_S = 0.9·R` (`R` the 90° rotation), `G_N = diag(0.1, 0.9)`, `e_0 = (1, 0)`: both maps are 0.9-Lipschitz, zero-preserving and invertible, yet

| quantity | value |
|---|---|
| `h(N) = ‖G_N e_0‖²` | 0.010 |
| `h(S ⊕ N) = ‖G_N G_S e_0‖²` | 0.656 |

Consequences. Theorem 4.1's hypotheses are established only for commuting (simultaneously diagonalisable) ensembles, where postfix monotonicity holds directly from Lemma B.4. For every non-Jacobi solver in the study (GS, SOR, SymGS, multigrid) the error propagation maps do not commute with the corrector (measured commutators 1e-2 to 1e-1, Table `tab:ca_assump`), so the theorem is not shown to apply. In addition, the main text's sentence "Invertibility of the error propagation functions is often satisfied with Jacobi and Gauss-Seidel updates" (line 418) is contradicted by the appendix's own measurements: `σ_min = 0` for GS, SOR and SymGS, and undamped Jacobi annihilates the mode `(N/4, N/4)`.

### T2 (major for the proof, minor for the result). The weak-supermodularity bound uses an invalid "reverse triangle" step, and the definition does not match its use

Step (a) of the upper bound asserts `‖x‖² − ‖G_O x‖² ≤ ‖x − G_O x‖²`. This is false: with `G_O = 0.5·I` the left side is `0.75‖x‖²` and the right side `0.25‖x‖²`. The final inequality survives because `h(S) − h(S ⊕ O) ≤ h(S)` trivially (as `h ≥ 0`), which even gives the sharper constant `α(O) = max{1/(T − Σρ²), 1}` instead of `4/(T − Σρ²)`. The proof should be replaced by the one-line argument, and the constant updated.

Two further mismatches: the definition of weak supermodularity quantifies over `S ∈ Ω^{|S'|}` (equal length), but the proof of Theorem 4.1 applies it with `S = S^t` of length `t < T`; and the final display of the proof of Proposition 4.2 has `T·max_i` in the denominator while the definition uses `Σ_i`. The result is fine with `Σ_i` because the lower bound was derived for the sum, but the definition must be changed to arbitrary-length `S` for the theorem's proof to go through.

### T3 (major, about what the theorem says). The guarantee is weaker than the trivial contraction bound in every regime studied

Theorem 4.1 bounds the *error reduction*: `g(∅) − g(S^T) ≥ (1 − φ_T(α))(g(∅) − g(O))`. With `α = 1` and `T = 300` this is `g(S^T) ≤ 0.63·g(O) + 0.37·g(∅)`, i.e. at best a constant-factor reduction of the initial error, independent of `T`. Any single convergent solver applied `T` times already gives `g ≤ ρ^{2T} g(∅)`; with `ρ = 0.99` and `T = 300` that is `0.0024·g(∅)`, 150× stronger than the theorem. The phrase "constant-factor approximation to the optimal strategy" in the abstract and contributions is technically about the reduction, but a reader will take it as a bound on the final error, which the theorem does not provide.

The paper's own assumption check makes this worse: the `α(O)` of Proposition 4.2 evaluated along the oracle's path is 1 for the multigrid pairing but 10²–10⁴ for every point smoother and infinite on 1–2 of 8 instances (Table `tab:ca_assump_paths`, macro `\caAlphaBoundMax` = 17652). With `α = 100`, `1 − e^{−1/α} ≈ 0.01`: greedy is guaranteed 1% of the optimal error reduction.

### T4 (major). The norm in which the assumptions are verified is not the norm the algorithm optimises

Appendix E.7 reports Euclidean Lipschitz constants above 1 for GS (1.02), SymGS (1.01) and SOR (1.45 on Poisson, 1.89 on anisotropic diffusion) and argues that the proofs "hold verbatim with ‖·‖₂ replaced by any norm", then verifies contraction in the energy norm of the symmetric part. That is correct for the proofs, but the objective `h`, Algorithm 1, the oracle in `hybrid.py` (`run_untimed`, policy `oracle`), and the router labels in `router.py` all use the Euclidean norm. A theorem for an energy-norm greedy rule does not cover the Euclidean greedy rule that was run. For convection–diffusion the "energy norm of the symmetric part" is additionally not the natural norm for the nonsymmetric operator.

### T5 (moderate). Proposition 4.3 and its worked example

Proposition 4.3 states `I − C_j L = P Λ_j P^{-1}` with no condition on `P`, but Lemma B.4 needs `P` orthogonal (unitary) to drop it from the norm. The main-text example "Jacobi, Gauss-Seidel, and a single-layer linear FNO" share an eigenbasis is wrong for lexicographic Gauss-Seidel on a periodic grid: the sweep is not translation invariant, and the paper's own commutator with the Fourier-diagonal corrector is 3e-2, not 0. For the corrector, `ρ = 1` (identity off-band), so Proposition 4.2 excludes it and only Proposition 4.3 can apply; the shared-eigenbasis condition is then verified only numerically (commutators ~1e-6, i.e. float32 round-off), which is evidence but not the exact hypothesis.

### T6 (moderate). The "empirical supermodularity ratio" check is uninformative

`check_assumptions.py:396-407` computes `α̂` with `S` the oracle's prefix and `S'` the oracle's *own suffix*, and takes the maximum over prefixes. At the last prefix (`|S'| = 1`) the ratio is identically 1, so the reported "`α̂ ≤ 1.000` in every setting" is attained trivially, and for long suffixes the ratio is automatically small because the denominator sums `|S'|` single-step gains. The definition requires the inequality for *all* `S` and with `S' = O` (the optimal sequence, which is unknown). The sentence "the objective behaves supermodularly on the states the greedy rule actually visits, which is the regime in which Theorem 4.1 gives the (1 − e^{-1}) guarantee" is not supported by this diagnostic.

### T7 (moderate). The cost-aware macro-action remark over-claims what transfers "verbatim"

For actions cheaper than the unit, `m_j` repeats form a stationary iteration and the remark is right. For actions dearer than the unit (multigrid at every grid, SymGS at 256² and 512²), `Env.macro_score` compares `exp_j · log(‖e_j‖/‖e‖)` with `exp_j = u/(m_j c_j) < 1`. This is a rate heuristic (geometric extrapolation of one step), not Algorithm 1 on a preconditioner set, and a horizon of `T` such decisions is not a budget of `≈ T u` (at 512² one multigrid cycle costs 7.6 units). The paper's statement that the five stationary solvers are cost-equalised "to within 8%" is also false; from `checkpoints/costs_*.json`:

| grid | Jacobi | dJ | GS | SymGS | SOR |
|---|---|---|---|---|---|
| Poisson 128² (`m·c_j/u`) | 0.98 | 0.97 | 0.93 | **1.15** | 0.93 |
| Poisson 256² | 1.00 | 0.93 | **0.77** | **1.51** | **0.80** |
| Poisson 512² | 0.96 | 0.86 | 1.06 | **1.78** | 1.07 |
| ConvDiff 512² | 1.18 | 0.81 | 1.08 | **1.83** | 1.08 |

The exponent correction is applied to these cases in the code, so the oracle remains cost-aware, but the text should say so and drop the 8% figure.

### T8 (minor). Theorem 5.1 and proof hygiene

The Bayes-consistency proof (Appendix C.3) is correct as a per-state, cost-sensitive calibration argument built on Mao–Mohri–Zhong. Two things should be said plainly: the labels are a deterministic function of the state under teacher forcing, so with the class of all measurable functions the statement reduces to "a rich enough classifier can fit a deterministic labelling"; and it says nothing about sequence-level error, distribution shift under the router's own rollouts, or time-to-tolerance. Typos that impede verification: "(a) by μ-postfix monotonicity" is a leftover from a removed definition; the rearranged inequality in the proof of Theorem 4.1 has a sign error in the intermediate line (correct in the final line); the proof of Proposition 4.3 writes `h(S') − h(S ⊕ ω)` for `h(S') − h(S' ⊕ ω)`; Theorem 4.1 defines `O` through `h` while the statement is about `g`.

---

## 3. Experiments: correctness findings

### E1 (major). The deployed corrector is a linear least-squares fit of the exact band-limited inverse, not a neural operator

`corrector.py` defaults (`--mlp 0`, `--trunk fourier`, `--linear_fit ls`, lines 483–488) give: FFT restriction to the band `|k| ≤ 31`, one dense `4096 × 3969` matrix fitted by ridge regression on 32k–64k exact `(residual, error)` pairs of the discrete operator (which is diagonal in exactly this basis), expansion in the fixed orthonormal Fourier basis, FFT prolongation. The model has no nonlinearity; the "scale-equivariant application" `‖Rr‖·G(Rr/‖Rr‖)` is the identity for a linear `G`; the reported validation error of 2e-6 is float32 round-off. The appendix describes this honestly, but the abstract, introduction and related-work discussion of spectral bias, and the term "DeepONet corrector" in Section 7, describe something else. Every statement about "neural operators" in the contributions should be re-scoped.

This also changes the baseline landscape. The corrector already performs a full-size `rfft2` and `irfft2` on every call (`BandTransfer.restrict/prolong`); dividing by the symbol on the band inside that FFT would implement the identical operator without the `4096²` matvec. The measured FFT direct solve (`results/baselines_*.json`) is faster than every router in every cell:

| setting | FFT solve | best router at `h²` |
|---|---|---|
| Poisson 128² | 0.32 ms | 0.91–1.13 ms |
| Poisson 256² | 1.32 ms | 2.38–4.22 ms |
| Poisson 512² | 7.65 ms | 38.7 ms ({NO, MG}); 60.7–131.5 ms (stationary) |
| ConvDiff 128² | 0.38 ms | 1.10–1.59 ms |
| ConvDiff 512² | 8.06 ms | 53.7 ms ({NO, MG}) |

The protocol paragraph declares FFT "not a competitor". For a constant-coefficient periodic problem it is *the* competitor, and it should be in Table 1. If the intended scope is problems where FFT is unavailable, the corrector as designed (fixed Fourier trunk, exact FFT transfers, exact pairs of the same operator) does not transfer to them either, so the paper currently has no experiment in its intended scope.

### E2 (major). The HINTS baseline in the code differs from the paper's definition of HINTS, and the difference is the headline effect

Section 3 defines HINTS by `S_t = 1[t mod τ > 0] + 1`, which with the paper's indexing (`u^{(t+1)} = u^{(t)} + C_{S_t}(...)`, `t ≥ 0`) selects the neural operator at `t = 0`. The implementation (`hybrid.py:246-247`, and the same phase in the old `HINTSRouter`, `hybrid_solver.py:55`) selects it when `(it + 1) mod τ == 0`, i.e. first at iteration τ, after τ−1 sweeps. From `results/Poisson_128_jacobi.json`: the router reaches `h²` at iteration 2 (one corrector call, one sweep); HINTS-25 at iteration 25; HINTS-5 at iteration 5. Work-unit arithmetic with the paper's own costs reproduces the reported medians exactly (router 0.74 ms, HINTS-25 1.92 ms, HINTS-5 0.89 ms), and shows that the entire margin at `h²` is the τ−1 sweeps spent before the first call. A HINTS variant that calls the corrector at `t = 0` (the paper's own formula), or `τ ∈ {1, 2}`, is the fair comparison and is absent; the "best τ" search covers only `{5, 10, 25, 50}`. The same phase effect produces the "orders of magnitude" AUC gap (HINTS-25 spends 24 iterations at relative error ≈ 1, so its AUC ≈ 24).

### E3 (major). Wall-clock margins are confounded by the efficiency of the stencil implementation

The paper argues that `O(N²)` numpy stencils and cached sparse triangular solves mean "the classical baselines are not artificially slowed". Asymptotic order is not the issue; constants are. The corrector is a single BLAS `sgemv` (16.7 MFLOP in 0.63–0.76 ms, i.e. near peak), whereas a numpy Jacobi sweep at 128² is 6 µs of update plus 47 µs of `np.roll`-based residual. I compiled a plain C sweep (`cc -O3`, single thread, no SIMD tuning) and timed it on the same machine in the same session:

| N | C fused Jacobi sweep | numpy Jacobi update + residual | C GS sweep | scipy `splu` GS solve (+ residual) | corrector (same session / cached) |
|---|---|---|---|---|---|
| 128 | 10.8 µs | 6.3 + 47.3 = 53.6 µs | 52 µs | 162 (+47) µs | 758 / 635 µs |
| 256 | 41 µs | 20.5 + 180 = 200 µs | 237 µs | 652 (+180) µs | 977 / 1004 µs |
| 512 | 169 µs | 170 + 1007 = 1177 µs | 1006 µs | 2907 (+1007) µs | 2730 / 3363 µs |

Consequences, using the paper's decision sequences (which are correct and unchanged) and 15 µs per Jacobi sweep including a fused stopping test at 128²:

- The corrector costs ≈ 45–60 Jacobi sweeps, not 12; macro-action sizes, and hence the oracle's decisions, change.
- Poisson 128² Jacobi at `h²`: router ≈ 0.67 ms, HINTS-25 ≈ 1.01 ms (1.5×, reported 2.68×), HINTS-5 ≈ 0.71 ms (1.06×, reported 1.21×). The "faster than the best period in every cell" claim is at parity for the Jacobi pairings.
- Multigrid: a V(2,2) cycle at 128² with the C GS sweep is ≈ 0.35 ms (four fine-level sweeps, transfers, coarser levels, 32² FFT), so three cycles ≈ 1.0 ms, equal to the router's 0.99 ms; the reported 3.6–3.7× advantage over multigrid disappears. At 512² four cycles ≈ 22 ms versus 38.7 ms for the {NO, MG} router and 61–132 ms for the stationary routers: multigrid alone would be the fastest method. The abstract's "faster than geometric multigrid and multigrid-preconditioned Krylov methods" does not survive a compiled smoother, and it is already not true at 512² for the stationary pairings in the paper's own Table `tab:ca_speed_poisson` (GS 0.69×, Jacobi 1.14× at p = 0.12).

The comparison would be fair only if all components were implemented at comparable efficiency (e.g. all in compiled code, or the corrector also in numpy-speed code), or if the paper reported results as a function of the corrector/sweep cost ratio.

### E4 (major). The validation claim "match the dense implementations to 1e-12" is false for SymGS, because the dense SSOR at HEAD is wrong

Running `validate_fast_pde.py` gives one-step discrepancies of 1.5e-12 to 6.6e-12 for Jacobi, damped Jacobi, GS and SOR, but **8.0e-4 (Poisson) and 4.9e-4 (ConvDiff) for SymGS**; the script has no assertions and prints `done`. The cause is `numerical_solver.py:195` at HEAD: `second_term = ω/(2−ω) + D⁻¹` adds a scalar to every entry of `D⁻¹` instead of multiplying (`FastSSOR` in `fast_pde.py` implements the standard formula). Wall-clock SymGS rows are therefore fine, but the Dirichlet table (`tab:errorcomparison_dirichlet`), produced by the old pipeline, used the wrong preconditioner in its "SymGS" rows, which is consistent with SymGS-only being worse than Jacobi-only there. (An uncommitted fix to this line appeared in the working tree during the audit; the Dirichlet SymGS rows still need re-running.)

### E5 (major). The manuscript is out of sync with the results, and results are still being produced

Regenerating `paper/costaware_tables.tex` from `results/` changes 108 lines. Three 512² cells (Poisson damped Jacobi, Poisson SymGS, ConvDiff damped Jacobi) appear that are not in the PDF; `\caNumCellsC` goes from 6 to 9 and `\caSpBestCMin` from 1.09× to 1.08×. In the new ConvDiff 512² damped-Jacobi cell the router's median time (222.2 ms) is *larger* than HINTS-25's (203.4 ms) while the paired median ratio is 1.08× in the router's favour and the cell is bold; the caption's "bold: faster than both" therefore needs a single, stated estimand (median of paired ratios vs ratio of medians), and both should be shown where they disagree. The ConvDiff 512² SymGS cell is being computed at audit time. The three new JSON files are untracked, contradicting the README's "all reported numbers are tracked".

### E6 (moderate). Configuration choices made after looking at the test set

Disclosed in the text but still test-driven: (a) "best τ" chosen per cell on the test set (deliberately optimistic, fine if labelled); (b) the ConvDiff 512² Jacobi router retrained with 64 rollouts and 800-iteration horizons after the first router missed `h²` on 4 of 16 *test* instances; (c) the ensemble recipe (256 rollouts, 6 DAgger rounds, width 128, 300 epochs) adopted after observing a gap on a named test cell (ConvDiff 128², 1e-8, 3.34 vs 2.67 work-unit ms); (d) the corrector band `|k| ≤ 31` and the anisotropic `N × N/4` sensor grid were chosen knowing that they make one call sufficient for `h²`. The README sentence "Nothing is tuned on the test set except the best fixed τ" is inaccurate. A held-out validation seed for every such choice, followed by a single evaluation on fresh test seeds, is required for the "all cells" claims to be confirmatory.

### E7 (moderate). The "final error after T iterations" column is an artefact of the stopping rule

`bench.py:40` stops every untimed run at true relative error 1e-9 and `bench.py:214` pads the trajectory with the last value. The router typically crosses 1e-9 with a sweep (final ≈ 7e-10) while HINTS often crosses it with a corrector call (final ≈ 1e-12), so Tables `tab:ca_auc_*` show the router's "error after T = 300 iterations" 100–1000× above HINTS-25's in four of five pairings (e.g. damped Jacobi: 7.57e-10 vs 1.15e-12). Neither number is the error after 300 iterations. Either run the full horizon for this metric or drop the column.

### E8 (moderate). Statistics and reporting

- `results.py:561` writes `np.std`, but the Dirichlet table caption says "standard error (s.e.)"; the values are standard deviations (the caption also says 128 instances where the main text says 64).
- One timed replay per instance (`--timed_reps 1`); 16 instances at 512²; margins of 5–11% with one-sided Wilcoxon tests and no multiplicity control across ~150 tests.
- Censored runs enter the paired tests at their time-to-cap (`make_tables.py:82`); conservative for a superiority claim, but the resulting p-values are not a censored-data analysis.
- Costs are the minimum over blocks of block medians (`measure_costs`), i.e. a favourable lower envelope, used both for macro sizes and for all work-unit numbers.
- The text says the cost-agnostic greedy "makes the same decisions" as the cost-aware oracle at 128²; on the first stored instance it makes its second corrector call at iteration 3 versus 14. The statement holds only for the path to `h²`.

### E9 (moderate). Claims in the abstract not supported at the largest grid

"It beats HINTS on every grid even when the latter's period is tuned per setting" holds by the paired-ratio estimand but not by median times in the new ConvDiff 512² damped-Jacobi cell (E5). "Faster than geometric multigrid and multigrid-preconditioned Krylov methods on the isotropic equations" is true at 128² and 256² only; at 512² the stationary routers are slower than or statistically indistinguishable from multigrid alone (E3), and the claim additionally depends on the sweep implementation.

### E10 (minor). The `h²` "truncation level" is asserted, not shown

The forcing distribution includes `γ = 0.5` fields, for which the discretisation error of the 5-point scheme is not obviously `O(h²)` with a constant near 1 in relative `ℓ₂`. A nested-grid comparison would either justify the tolerance or reduce it to a conventional grid-dependent target.

### E11 (checked and found correct)

For completeness, the following were verified and are sound: the oracle scores actions by the log ratio of error after/before, so it is scale-invariant; the router sees only residual-derived features (no leakage of the true error); the timed replay charges the router's feature and decision cost and one residual per operation for every method; `work_units` matches the live protocol (initial residual once, one residual per operation; it does not double-count, contrary to `REVIEW.md` item 11); the surrogate in `router.py:188` is Eq. 10 with per-state normalised weights, whose argmin coincides with the oracle's; test/train/corrector seeds are distinct; the nested-ensemble table (7 of 12 wins, one 0.92× loss inherited from the oracle) and the seed table (identical decisions) match the result files.

---

## 4. Broader concerns (secondary)

1. **What is learned is a two-line schedule.** The oracle's policy is "corrector first, then smooth, corrector again only if the smooth error re-emerges"; decisions are identical across seeds and nearly identical across instances; ensemble routers agree with the oracle on only 47–83% of decisions, yet the paper's own control (Table `tab:ca_ens_big`) shows that those disagreements do not change time-to-`h²`. A hand-written "NO-first, then smoother" policy is the missing ablation; the learning contribution in this setting is thin.
2. **The premise of the original submission is gone.** The Limitations section still says routing pays off because "the neural operator exhibit[s] heterogeneous performance across samples". With an exact linear band solve there is no heterogeneity, which is why the router's decisions are instance-independent.
3. **This is a two-grid method.** Exact spectral coarse solve on `|k| ≤ 31` plus a point smoother is a two-grid cycle with a fixed coarse space; the loss of grid-independence at 512² is the textbook consequence. The anisotropic corrector (full resolution in `x₁`) encodes exactly the semi-coarsening knowledge a multigrid practitioner would use, but the corresponding line-relaxation/semi-coarsened multigrid baseline was dropped rather than run.
4. **Problem class.** All experiments are constant-coefficient, periodic, linear, and FFT-solvable. One experiment where the Fourier inverse is unavailable (variable coefficients, non-periodic boundaries at scale, or an irregular domain) is needed for the claims to have practical content; the Dirichlet appendix uses the old 31² pipeline and does not fill this gap.
5. **Reproducibility.** Checkpoints and cost caches are untracked; run scripts hard-code a personal interpreter path; JSON results carry no commit hash or environment record; the table generator consumes every matching JSON in `results/`, so stale or new files silently change the manuscript; the PDF was built from a dirty tree while a benchmark was running.
6. **Presentation.** The paper is a hybrid of two pipelines (dense/LSTM at 31² for Dirichlet; stencil/MLP at 128²–512² elsewhere), with Appendix C describing DeepONet and LSTM hyperparameters that now apply to one table. Red revision markup and a revision note remain in the submission file.

---

## 5. Claim-by-claim table

| Claim (location) | Status | Evidence |
|---|---|---|
| Prop. 4.2: invertibility ⇒ postfix monotone | **False** | 2-D counterexample, T1 |
| Prop. 4.2: `α(O) = 4/(T − Σρ²)` | True, proof step (a) invalid | T2 |
| Thm 4.1 applies to the evaluated ensembles | Not shown (non-commuting; Euclidean ρ > 1) | T1, T4 |
| "Constant-factor approximation to the optimal strategy" (abstract) | Misleading; weaker than trivial bound | T3 |
| "Invertibility … often satisfied with Jacobi and GS" (l. 418) | Contradicted by App. E.7 (σ_min = 0) | T1 |
| GS shares the Fourier eigenbasis (l. 430) | False (commutator 3e-2) | T5 |
| `α̂ ≤ 1.000` verifies weak supermodularity (App. E.7) | Trivially satisfied; wrong `S'` | T6 |
| Macro costs equalised "to within 8%" (App. E.1) | False (SymGS 1.15–1.83×; GS/SOR 0.77–0.80× at 256²) | T7 |
| Fast solvers match dense ones "to 1e-12" (App. E.1) | False for SymGS (8e-4); dense SSOR bug at HEAD | E4 |
| HINTS = `1[t mod τ > 0] + 1` (Sec. 3) | Code calls NO first at iteration τ, not 0 | E2 |
| Router 2.7–12× faster than HINTS-25 at `h²` (abstract) | Reproduced from files; driven by E2 and E3 | E2, E3 |
| Router faster than best τ in every cell | Reproduced by paired ratio; near parity with compiled sweeps; one cell slower by median | E3, E5 |
| Faster than multigrid and MG-Krylov (abstract) | True at 128²/256² only with numpy sweeps; false with compiled sweeps and at 512² | E3, E9 |
| "Classical baselines are not artificially slowed" | Not supported (4–5× slower than plain C) | E3 |
| Corrector is a "DeepONet"/neural operator | Linear least-squares band inverse | E1 |
| Values in Dirichlet table are s.e. | They are standard deviations | E8 |
| "All reported numbers are tracked" (README) | Three untracked result files; run in progress | E5 |
| Nested ensembles: 7/12 wins, one 0.92× loss | Confirmed | E11 |
| Decisions identical across 5 seeds | Confirmed | E11 |

---

## 6. Requests to the authors (ordered by importance)

1. Repair or narrow Proposition 4.2 and Theorem 4.1: state the result for simultaneously unitarily diagonalisable ensembles (where it is true), or supply a correct proof of postfix monotonicity under an additional hypothesis; fix the definition of weak supermodularity; replace step (a) and the constant 4; state explicitly what the bound guarantees relative to `ρ^{2T}`.
2. Evaluate the oracle and router in the norm in which the assumptions hold, or verify the assumptions in the Euclidean norm.
3. Describe the corrector as what it is (a linear band-limited inverse fitted by least squares) throughout, and add as baselines: the same correction computed inside the FFT, the full FFT solve, a conventional coarse-grid correction, and the static "corrector at t = 0, then smoother" schedule.
4. Implement HINTS as the paper defines it (first call at t = 0) and include `τ ∈ {1, 2}`; report how much of the gain survives.
5. Re-time all components at comparable implementation efficiency (compiled sweeps, or report results as a function of the corrector-to-sweep cost ratio), and re-derive macro-action sizes accordingly; re-state the multigrid comparison with a competently implemented V-cycle (and a line smoother for anisotropic diffusion).
6. Freeze every configuration choice on validation seeds and re-run the test evaluation once, with multiple timed replays, on fresh test seeds; regenerate all tables from a committed manifest after the running job completes; commit the new result files.
7. Fix the dense SSOR at HEAD, make `validate_fast_pde.py` assert, and re-run the Dirichlet SymGS rows; relabel the Dirichlet error bars.
8. Remove or re-run the "final error after T iterations" column (E7); define one speed-up estimand and use it consistently (E5).

---

## 7. Minor issues and typos

- "Gauss Elimination" → Gaussian elimination; "e.x." → e.g.; "autoregressivelt", "geomterically", "mild deviations", "suported", "distribtion", "the he risk" (App. C.3).
- Table `tab:routerhpsettings`: missing `&` in the `γ_tf` row.
- Section 2.2 mixes sampled `(a_i, f_i, u_i)` with the expectation over `(a, f)`.
- The introduction's "iterative methods … lack generalization" is a category error; they are algorithms, not predictors.
- Abstract: "consistently reduces final error and AUC … relative to single-solver baselines" is contradicted by the Dirichlet SOR/ConvDiff rows (p = 1.0, identical to solver-only).
- Cost-aware paragraph: `‖e‖^{u/(m_j c_j)}` should be `(‖e_j‖/‖e^{(t)}‖)^{u/(m_j c_j)}`, which is what the code compares.
- App. E.1: "The same corrector is used by HINTS" is right; say also that HINTS's corrector call count to 1e-8 (92 at Poisson/Jacobi) is what makes it slow there, not corrector quality.

---

## 8. Relation to the existing `REVIEW.md`

Confirmed independently: the postfix-monotonicity failure and the invalid inverse-Lipschitz step (its counterexample and mine differ; both are valid), the definition/usage mismatch, the missing orthogonality in Proposition 4.3, the SymGS validation failure, the stale tables (same three cells), the test-driven configuration changes, the FFT comparison, the `h²` justification gap, and the per-step nature of Theorem 5.1.

Added here: the invalid "reverse triangle" step and the resulting sharper constant (T2); the vacuity of the bound relative to `ρ^{2T}` and the paper's own `α(O)` values (T3); the HINTS phase discrepancy with the paper's own formula (E2); the quantitative implementation-efficiency confound with compiled measurements and its effect on the multigrid and best-τ claims (E3); the root cause of the SymGS mismatch in the dense code at HEAD (E4); the cost-equalisation numbers (T7); the final-error artefact of the stopping rule (E7); the std/s.e. labelling (E8); and the 512² statements in the abstract (E9).

Disagreement: `REVIEW.md` item 11 states that `work_units` double-counts the residual. It does not. `hybrid.py:362-372` charges the initial residual once at `t[0]` and then each executed operation's cost, which already contains its own residual, exactly as the live replay does. No regeneration is needed on that account.
