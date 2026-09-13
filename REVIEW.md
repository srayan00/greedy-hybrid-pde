# Conference-Style Review: *A Greedy PDE Router for Blending Neural Operators and Classical Methods*

## Summary and recommendation

This paper proposes a hybrid iterative PDE solver that chooses at each decision epoch between a learned residual corrector and one or more classical iterations. The paper studies an oracle greedy rule, proposes a cost-sensitive cross-entropy surrogate for imitating that rule, and introduces a wall-clock implementation based on cost-equalized macro-actions and a small scalar-feature MLP router. Experiments cover periodic constant-coefficient Poisson, convection--diffusion, and anisotropic diffusion problems on grids from $128^2$ to $512^2$, with comparisons to fixed-period HINTS, stationary iterations, multigrid, and Krylov methods.

My overall recommendation in the paper's current state is **reject / major revision**. The empirical phenomenon is real and potentially useful: on the supplied instances, an initial band-limited correction followed by a suitable smoother is much faster than delaying the correction until a fixed HINTS period, and the learned router generally imitates the cost-aware oracle accurately. The repository is unusually extensive, and the authors make a serious effort to account for wall-clock costs, censoring, training costs, seeds, and solver ensembles.

However, the principal theoretical approximation guarantee is not established as written, and the paper overstates the degree to which its assumptions are verified for the experimental algorithm. There are also material experimental-validity and reproducibility concerns: a solver-equivalence validation fails for SymGS; the newest result files and manuscript tables are out of sync; some reported configurations were selected after observing performance on the same test set; timing uses only one repetition in the primary runs; and the method is not compared prominently against the FFT direct solver that is both available for these test problems and substantially faster in the supplied measurements. These are correctness issues rather than matters of exposition.

**Suggested score:** 3/10 (reject), with reasonably high confidence. I would be open to a substantially revised paper centered on the empirical scheduling result, provided the theorem is repaired or narrowed and the evaluation is regenerated under a locked, auditable protocol.

## What I inspected

I reviewed the current working tree, including the uncommitted manuscript edits and the experiment outputs produced on September 13, 2026. I inspected the main paper and cost-aware appendix, the fast PDE and solver implementations, corrector, router, rollout/timing code, strong baselines, assumption checker, table generator, run scripts, JSON results, and checkpoint metadata. I also:

- ran `validate_fast_pde.py` in the documented `ansatz` environment;
- compiled the Python sources;
- independently regenerated `costaware_tables.tex` to a temporary path and compared it with the manuscript's current table file;
- checked result counts, configurations, baseline timings, assumption outputs, and Git tracking/ignore state;
- constructed a two-dimensional counterexample to the postfix-monotonicity argument used by the main theorem.

I did not retrain every corrector/router or repeat the multi-hour benchmark suite. Thus, this is a source-and-artifact audit with targeted executable checks, not a full clean-room reproduction.

## Strengths

1. **The core scheduling observation is clear and well supported within the chosen setup.** A fixed schedule that waits 24 classical iterations before the first corrector call is predictably inefficient when the corrector is most useful at the initial, smooth-error state. The supplied trajectories consistently show that the learned router discovers the simple and effective pattern of calling the corrector immediately and then using classical smoothing.

2. **The implementation is substantially more careful than the original dense-grid pipeline.** `fast_pde.py` uses stencil operations and cached sparse triangular solves rather than dense $N^2\times N^2$ matrix products. The benchmark separates untimed error tracing from timed replay and charges the learned router for feature extraction and inference.

3. **The paper includes many useful controls.** It reports multiple HINTS periods, paired tests, five training seeds at $128^2$, solver ensembles, oracle agreement, operation costs, training amortization, larger grids, and an anisotropic model problem. The ensemble section also candidly reports a case where adding an action hurts the greedy policy.

4. **The cost-sensitive surrogate has a sound basic target.** For costs $c_j$ and weights $w_j=\sum_{k\ne j}c_k$, maximizing the optimal softmax probability is equivalent to minimizing $c_j$. The code implements this logic after a per-state normalization, which does not alter the Bayes decision.

5. **The authors expose limitations more honestly than is typical.** The appendix records non-normal Euclidean expansion, non-invertibility, the corrector's unit Lipschitz constant outside its band, test-set tuning of the optimistic HINTS baseline, and degradation when the fixed corrector band does not scale with the grid.

## Major correctness concerns

### 1. The main greedy approximation theorem is not proved under the stated assumptions

Theorem 4.1 needs postfix monotonicity: for arbitrary sequences $S,N$, the paper needs $h(S\oplus N)\le h(N)$. Proposition 4.2 claims that invertibility of every error-propagation map, together with a Lipschitz constant below one, implies this property. The proof inserts $G_N^{-1}G_N$ and then effectively bounds the inverse by the reciprocal of the forward Lipschitz constant. This implication is false: an upper bound $\|Gx\|\le\rho\|x\|$ supplies no upper bound $\|G^{-1}y\|\le\rho^{-1}\|y\|$. The latter would require a lower singular-value bound.

More fundamentally, even invertible strict contractions need not be postfix monotone when they do not commute. For example, let

```text
A = [[ 0.0674240, -0.1864331],
     [-0.1473150, -0.8707252]]
B = [[ 0.7575738,  0.4816284],
     [-0.1369844,  0.3257283]]
x = [0.2812107, -0.5538228].
```

Both matrices are invertible and have spectral norm $0.9$, yet
$\|BAx\|=0.3302>0.2254=\|Bx\|$. Taking $S=(A)$ and $N=(B)$ directly violates postfix monotonicity. Therefore Proposition 4.2 and the invocation of Theorem 4.1 do not apply to the general noncommuting ensemble claimed by the paper.

This is central: the abstract and contributions present the constant-factor approximation as a principal theoretical result. A repair would need stronger structural assumptions (for example, appropriate simultaneous orthogonal diagonalization/commutation), a different notion of monotonicity, or a new proof and bound.

### 2. The definition of weak sequence supermodularity does not support the theorem proof as written

The paper defines weak supermodularity with respect to $S'$ for `any` $S\in\Omega^{|S'|}$. In the proof of Theorem 4.1, this property is applied with $S=S^t$, whose length is $t$, and $S'=O$, whose length is $T$, for every $t<T$. That use is outside the stated definition. If equal length was not intended, the definition must be changed and Proposition 4.2 reproved for the broader quantification. If it was intended, the theorem proof is invalid.

There are related formal inconsistencies: the theorem defines $O=\arg\min h(S)$ even though the generic objective is named $g$; the final weak-supermodularity display in the proof switches between a sum and $T\max_i$; and the simultaneous-diagonalization proof contains an incorrect left-hand expression (`h(S') - h(S \oplus \omega)` where `h(S') - h(S' \oplus \omega)` is needed). Some are likely typographical, but together they make the main result difficult to verify.

### 3. Proposition 4.3 is missing the orthogonality/normality condition its proof requires

The proposition states $G_j=P\Lambda_jP^{-1}$ with a common $P$ and $\|G_j\|\le1$, then concludes Euclidean-norm supermodularity. The preceding lemma and its proof explicitly require $P$ to be orthogonal in order to remove it from the Euclidean norm. A merely invertible shared eigenvector matrix is insufficient. The proposition must require simultaneous **orthogonal/unitary** diagonalization (or change the norm to one induced by $P$).

The prose also gives Gauss--Seidel as an example of an operator sharing the Fourier eigenbasis on periodic constant-coefficient problems. Lexicographic Gauss--Seidel is not generally circulant/FFT-diagonal because the ordering breaks translation symmetry. The repository's own commutator measurements are nonzero for GS, SymGS, SOR, and multigrid.

### 4. The experimental assumption check does not establish the theorem for the algorithm actually evaluated

The algorithm and reported error objective use the Euclidean relative error. For GS, SymGS, and SOR, the measured Euclidean operator norms exceed one (up to approximately 1.89), so Proposition 4.2's contraction premise fails. The paper responds that the maps contract in an energy norm and that the proof works in any norm. Even if that statement were accepted, it proves a guarantee for an energy-norm objective and an energy-norm greedy rule. The experimental oracle and learned labels minimize Euclidean error, not energy error. A theorem about a different greedy decision rule does not validate the evaluated algorithm.

The corrector also has norm exactly one because it leaves all out-of-band modes unchanged, so the strict-contraction proposition does not include it. The paper instead invokes simultaneous diagonalization for combinations with (damped) Jacobi, but the learned least-squares corrector is only approximately Fourier diagonal: measured commutators are around $10^{-6}$ rather than exactly zero. Approximate commutation is not covered by Proposition 4.3. Numerical measurements on random vectors cannot establish exact shared eigenvectors or a worst-case theorem premise.

Finally, the reported empirical $\hat\alpha$ is evaluated only on greedy-oracle prefixes and that same path's suffix. This is an interesting diagnostic, but it is not the uniform weak-supermodularity property needed in the theorem, and it does not involve the unknown optimal sequence $O$. The statement that this places the experiments in the regime where the theorem gives the $(1-e^{-1})$ guarantee is therefore unsupported.

### 5. The cost-aware macro-action construction is not covered by the equal-cost theory in important cells

For cheap operations, rounding $u/c_j$ produces approximately equal-cost macro-actions. For an operation more expensive than the corrector, however, the implementation applies it once and scores it using
$\log(\|e_j\|/\|e\|)u/c_j$. This is a geometric-rate extrapolation; it is not literally the immediate error reduction per unit cost and it does not turn the action into one costing $u$. At $512^2$, for example, a multigrid cycle costs roughly $21.7$ ms while the Poisson corrector costs roughly $2.9$ ms. A horizon of $T$ such decisions plainly does not correspond to approximately $Tu$ wall-clock budget.

Consequently, running the fixed-horizon greedy theorem on these unequal-cost choices does not provide the stated wall-clock approximation. The score may be a reasonable heuristic, but the paper should call it that or supply a proper variable-cost/knapsack-style analysis. The surrogate theory also concerns the chosen per-step costs, not end-to-end time-to-tolerance optimality.

### 6. The paper's solver-equivalence claim is contradicted by its validation script

The paper says all fast one-step iterates match the dense implementations to $10^{-12}$. Running the provided validation gives approximately:

| Equation | Method | maximum one-step absolute discrepancy |
|---|---:|---:|
| Poisson | Jacobi / damped Jacobi / GS / SOR | $1.5\times10^{-12}$ to $6.6\times10^{-12}$ |
| Poisson | SymGS | **$8.0\times10^{-4}$** |
| ConvDiff | Jacobi / damped Jacobi / GS / SOR | $5.6\times10^{-9}$ to $3.7\times10^{-8}$ |
| ConvDiff | SymGS | **$4.9\times10^{-4}$** |

Thus the claim is false, and the script has no assertions, so it prints `done` despite this failure. Inspection suggests the older dense `SymmetricSuccessiveOverRelaxationSolver` itself may implement the SSOR middle factor incorrectly, while `FastSSOR` is closer to the standard formula. That may exonerate the fast solver mathematically, but it does not rescue the stated equivalence. The reference must be corrected, the comparison rerun with relative and absolute tolerances, and the validation made test-failing.

### 7. Current manuscript tables are stale relative to the current results

Regenerating tables to a temporary file changes the $512^2$ Poisson and convection--diffusion rows by adding newly produced damped-Jacobi and SymGS cells. The checked-in `paper/costaware_tables.tex` reports six $512^2$ stationary cells, whereas regeneration from the current results reports nine. This matters because the prose says the router wins in “all” stationary pairings.

One newly generated convection--diffusion/damped-Jacobi cell is especially awkward: the displayed median router time is about 222.2 ms versus 203.4 ms for HINTS, while the paired median of per-instance ratios favors the router by 1.08x. Both statistics can coexist, but a table caption saying bold means “faster than both” is ambiguous or misleading when the displayed medians have the opposite ordering. The paper must define whether “faster” means ratio of medians or median paired ratio and use one estimand consistently.

The newly generated JSON is ignored by Git and is not part of the tracked result set. A clean checkout therefore cannot recreate the current manuscript table without rerunning a long experiment.

## Major experimental-design concerns

### 8. Some hyperparameters were adapted to performance on the reported test set

The paper explicitly says that the first $512^2$ convection--diffusion/Jacobi router missed the target on four of 16 test instances, after which it was retrained with more rollouts and a longer horizon and reevaluated on the same 16 instances. The ensemble router budget was likewise increased after observing a gap on a named test cell. These are test-driven model-selection decisions. The resulting test statistics and “all cells” claims are no longer confirmatory, even though the changes are disclosed.

A validation split should be used to choose training horizon, number of rollouts, DAgger rounds, width, corrector band, and any per-cell exception. The final configuration should then be frozen and evaluated once on fresh test seeds. The best-HINTS-on-test baseline is deliberately optimistic and is acceptable when labeled, but it does not justify tuning the proposed method on that same set.

### 9. The practical comparison omits the strongest method available for the chosen problems

Every evaluated PDE is linear, constant-coefficient, and periodic, so it is exactly diagonalized by FFT. The code measures this direct solver, and the result files show median times to $h^2$ of approximately:

| Equation/grid | FFT | Representative best router in main table |
|---|---:|---:|
| Poisson $128^2$ | 0.32 ms | about 0.9--1.1 ms |
| Poisson $256^2$ | 1.32 ms | about 2.4--4.2 ms |
| Poisson $512^2$ | 7.65 ms | about 39 ms with NO+MG |
| ConvDiff $128^2$ | 0.38 ms | about 1.1--1.6 ms |
| ConvDiff $256^2$ | 1.56 ms | about 4.8--7.1 ms |
| ConvDiff $512^2$ | 8.06 ms | about 49 ms with NO+MG |

The manuscript declares FFT to be a reference rather than a competitor. That is an artificial distinction from a practitioner's perspective: for exactly these PDEs, FFT is the natural strong baseline and is substantially faster. It should appear in the headline table and discussion. If the intended contribution is for problems where FFT is unavailable, the evaluation needs variable coefficients, irregular geometry, nonperiodic boundaries at scale, nonlinear PDEs, or another setting where the corrector cannot itself exploit exact Fourier diagonalization.

### 10. The “DeepONet” corrector is extremely close to a problem-specific spectral inverse

The wall-clock corrector uses ideal FFT restriction, a fixed complete Fourier basis, exact FFT-generated residual/error targets, and a dense linear branch fitted by least squares. For a constant-coefficient periodic operator, the inverse on that band is analytically diagonal in precisely this basis. The resulting method is more naturally viewed as a learned approximation to a known band-limited spectral inverse than as evidence for general neural-operator/classical-solver routing.

This does not make the experiment invalid, but it changes its scientific meaning. The speedup largely demonstrates that applying a highly accurate coarse spectral correction immediately is better than waiting until iteration 25. A decisive ablation would compare against the exact band-limited Fourier inverse, a conventional coarse-grid correction, and a simple hand-written schedule (“corrector once at iteration 0, then smoother”). Based on the reported action traces, that static one-shot policy may match the learned router at $h^2$, weakening the claim that state-dependent learned routing is needed for the headline result.

### 11. Wall-clock methodology is too fragile for several small claimed margins

Primary benchmark runs use `timed_reps=1`, and many reported differences at $512^2$ are only 5--10%. Cost caches use a minimum-over-block-medians estimator, which estimates a favorable lower envelope rather than typical deployed latency. Results are collected in multiple sessions, and the repository itself notes contention in some runs. The rotating execution order helps, but one replay per instance is inadequate to support small wall-clock margins.

The benchmark also does not apply identical stopping-check overhead to all methods despite saying it does. Hybrid timed replay explicitly computes the new residual and its norm every operation. Timed Krylov calls run SciPy with no callback and do not execute the benchmark's explicit per-iteration true-error or stopping calculation. SciPy algorithms have their own internal residual logic, but this is not “the same residual evaluation of the stopping test in every iteration.” Comparisons should either time solvers to a residual tolerance in their normal optimized implementations, or instrument all methods under a genuinely common protocol and document the overhead.

There is also a small accounting bug in `work_units`: operation costs already include residual evaluation, but the cumulative array adds `_residual` both as the initial entry and again to every post-operation entry. Live timing does not appear to share this exact double count, so work-unit results and amortization values should be regenerated after correction.

### 12. The $h^2$ target is not demonstrated to equal discretization error for this data distribution

The paper calls relative algebraic error $h^2$ “the truncation level.” Second-order local truncation for a sufficiently smooth solution only implies an $O(h^2)$ discretization error with a problem-dependent constant and norm. Here the forcing distribution includes rough GRFs (including low regularity settings), and the target is relative discrete $\ell_2$ error against the exact solution of the same discrete system. No continuum or grid-convergence study is provided to show that the actual discretization error is approximately $h^2$ across instances and equations.

The target can still be used as a conventional grid-dependent tolerance, but the stronger statement that refinement below it is not meaningful needs evidence: evaluate a known continuum solution or compare nested-grid solutions and report the empirical discretization-error distribution.

## Statistical and reporting concerns

1. **Small sample sizes on larger grids.** Sixteen test instances, one timing replay, and one-sided tests are not persuasive for 5--10% margins. Confidence intervals for median paired log-speedups would be more informative than only p-values.

2. **Many comparisons.** The paper performs a large number of one-sided tests across equations, grids, solvers, tolerances, policies, ensembles, and seeds without multiplicity control. The very small p-values for large effects will survive correction, but marginal $p\approx0.005$ claims may not.

3. **Censoring is handled heuristically.** Replacing censored baseline times with the time at the iteration cap is conservative for a single superiority comparison, but a Wilcoxon test on these substituted values is not a principled censored-data analysis. Report success probabilities separately and use a survival/time-to-event method or a clearly labeled lower-bound analysis.

4. **Ratio estimands are mixed.** “Median speedup” is the median of paired ratios, while table bodies display medians of raw times. Their ordering can reverse, as in the new $512^2$ convection--diffusion/damped-Jacobi result. Choose and explain a primary estimand.

5. **Seed evidence is limited.** Five router-training seeds are provided only at $128^2$, while per-cell recipe changes and the smallest relative gains occur at larger sizes. Timing uncertainty and training uncertainty should both be measured where claims are most fragile.

6. **The Bayes-consistency result is per-step and on the teacher-forced/on-policy state distribution.** It does not imply consistency of the final sequence, time-to-tolerance, or approximation to the globally optimal schedule. The prose sometimes slides from “Bayes-optimal one-step imitation” to “recovering the guarantees” of the oracle sequence. This connection requires a sequential regret/distribution-shift argument that is absent.

## Reproducibility and repository concerns

1. `results/` is globally ignored even though many previously added result files remain tracked. New result files are therefore silently omitted. The README says all reported JSON numbers are tracked, which is no longer true.

2. Checkpoints are ignored, yet the advertised reproduction path requires large trained correctors and routers or many hours of retraining. The exact environment description in the README does not match `environment.yml` closely enough to ensure current package versions, and run scripts hard-code `/Users/yash/miniconda3/envs/ansatz/bin/python`.

3. The validation script prints errors but has no assertions and therefore succeeds despite a clear SymGS mismatch.

4. The result JSON stores arguments and instance hyperparameters, which is good, but lacks Git commit, code-dirty state, platform/package versions, checkpoint hash, start time, and run UUID. Current artifacts produced from a dirty tree cannot be tied unambiguously to the code and manuscript that generated them.

5. The primary table generator consumes every matching JSON in `results/`. This makes stale or newly added files silently change the manuscript. A submission manifest should enumerate exact artifact hashes used for each table and figure.

6. The paper keeps an older Dirichlet experiment produced by a different dense/LSTM pipeline. Its inclusion complicates the empirical story, and I did not find an equally explicit end-to-end reproduction manifest tying that retained table to a specific checkpoint and command. It should either be reproduced under the new protocol or clearly separated as historical evidence.

## Minor correctness and presentation issues

- The background writes the neural-operator risk using $u_i,a_i,f_i$ inside an expectation over $(a,f)$, mixing sampled and random-variable notation.
- “Gauss Elimination” should be “Gaussian elimination”; “e.x.” should be “e.g.”
- The introduction says classical iterative solvers “lack generalization,” which is a category error: they are algorithms, not learned predictors, and re-solving a changed right-hand side is not a failure to generalize.
- The cost-aware paragraph says all cheap macro-actions cost approximately $u$, but the text should report the actual maximum mismatch and distinguish those from costly one-step actions such as multigrid.
- The claim that the corrector “never injects high-frequency error” is true relative to the defined Fourier band because prolongation is band-limited, but it does not imply stability or error reduction within the band.
- Calling a fixed Fourier trunk with a linear least-squares branch a DeepONet is defensible architecturally, but the paper should disclose more prominently that the deployed map is essentially linear and Fourier-structured.
- The current manuscript contains a revision note and extensive red text. This may be appropriate for a revision response but should be checked against anonymization/submission-format rules.

## Questions for the authors

1. Can the authors provide a correct proof that invertible contractions imply postfix monotonicity for noncommuting error maps? If not, which theorem remains after restricting to simultaneously unitary-diagonalizable maps?

2. Which norm does the experimental oracle optimize? If Euclidean, how does an energy-norm contraction establish the assumptions for its decisions?

3. What policy is learned beyond the apparently sufficient static rule “apply the corrector at iteration 0, then use the smoother”? How does that baseline perform at $h^2$ and $10^{-8}$?

4. Why should a practitioner use the hybrid rather than the FFT solve on these exact constant-coefficient periodic problems? What happens when FFT diagonalization is unavailable?

5. Were the larger-router recipe, longer training horizon, anisotropic sensor geometry, and any other configuration choices selected before examining the reported test instances? If not, can the authors rerun on untouched seeds?

6. Why does `validate_fast_pde.py` report a $10^{-4}$-scale SymGS discrepancy while the paper claims $10^{-12}$ agreement?

7. Can the authors provide a continuum/grid-refinement calculation showing that relative algebraic error $h^2$ corresponds to the discretization-error floor for the sampled GRFs?

## Required changes for a credible revision

At minimum, I recommend the following before submission:

1. Repair or sharply narrow Proposition 4.2 and Theorem 4.1. Align the weak-supermodularity definition with its use, add the missing orthogonality condition to Proposition 4.3, and remove claims that numerical path diagnostics establish worst-case theorem assumptions.
2. State explicitly that surrogate consistency is one-step cost-sensitive classification consistency, not global sequence or wall-clock optimality.
3. Fix the SymGS reference/validation and convert all validation checks into assertions with documented relative tolerances.
4. Freeze the protocol, choose hyperparameters on a validation set, and rerun final evaluation on fresh test seeds with multiple timing repetitions.
5. Regenerate every table and figure from a committed manifest of exact JSON files; track the new results and record commit/checkpoint/environment hashes.
6. Add FFT, exact band-limited spectral correction, conventional coarse-grid correction, and the static one-shot-corrector schedule as baselines.
7. Reconcile paired-ratio speedups with raw median times and add uncertainty intervals plus multiplicity-aware inference.
8. Either justify $h^2$ empirically as a discretization floor or rename it a nominal grid-dependent algebraic tolerance.
9. Evaluate at least one problem where the Fourier inverse is not analytically available and where the learned corrector/router offers a genuine advantage over standard numerical structure.

## Bottom line

The repository supports a useful empirical lesson: when a powerful low-frequency corrector is available, applying it immediately and sparingly can beat a fixed periodic hybrid schedule by a large margin. The lightweight router successfully learns that schedule on the tested distribution. Those results are promising.

The current paper, however, claims substantially more: a general constant-factor greedy guarantee under assumptions that do not imply the needed postfix monotonicity, experimental verification of assumptions in a norm different from the optimized objective, and broad practical superiority on problems for which a measured FFT direct solve is faster. Combined with stale artifacts, test-driven configuration changes, and the failed SymGS equivalence check, these issues prevent acceptance on correctness grounds. A revision that treats the method as a carefully evaluated scheduling heuristic, repairs the theory, and broadens the numerical setting could become a strong contribution.
