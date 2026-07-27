# Fork DeepONet + ApproxGreedy lite router criteria experiment

Matches the **wallclock-rebuttal** corrector training, then trains our lite router
with the paper's ApproxGreedy loss and scores three criteria.

1. Scale-equivariant **DeepONet** (`train_fast_deeponet.py` — same as fork)
2. Residual-distribution fine-tune (`finetune_deeponet_residual.py` — same as fork)
3. **Lite scalar-feature MLP router** with **ApproxGreedyRouterLoss** soft weights
4. Wall-clock bench vs classical / HINTS / oracle

## Criteria

1. **Diverse routing** — instances use both Jacobi and NO; NO fraction varies
2. **Fewer iterations** than classical and HINTS (residual / error at \(h^2\))
3. **Less wall-clock** than classical and HINTS

## Run

```bash
sbatch batch/train_lite_fno_router.sbat
# optional: N=63 DON_EPOCHS=300 sbatch batch/train_lite_fno_router.sbat
```

Results: `playground/lite_criteria/results/criteria_poisson_N31_jacobi.json`
