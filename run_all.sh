#!/bin/zsh
# Master chain, ordered so that every 128^2 result (all four equations) is complete before any larger grid:
#   1. 128^2 pairwise cells (with same-session baselines) and ensembles of Poisson, ConvDiff, AnisoDiff
#   2. 128^2 variable-coefficient diffusion: corrector, pairwise cells, nested ensembles
#   3. re-timing pass of the 128^2 cells written by an earlier driver revision
#   3b. decision-granularity ablation for the pairings in which the router lost to the best schedule at 1e-8
#   3c. 256^2 Poisson and ConvDiff cells with their ensembles (brought forward for the same question)
#   4. 128^2 usage traces, seed trials and assumption / screening / theorem checks (all four equations)
#   5. decision-granularity ablation (128^2 only)
#   6. 256^2 and 512^2 cells with their ensembles and checks, overheads, discretisation study
# Every script skips steps whose output already exists, so the chain resumes after an interruption. Launch detached:
#   setopt NO_BG_NICE; nohup ./run_all.sh > logs/run_all.out 2>&1 &     (zsh otherwise starts background jobs at nice 5)
cd "$(dirname "$0")"
note() { grep -qF -- "$1" logs/final_progress.log || echo "$(date '+%F %T') ===== $1 =====" >> logs/final_progress.log; }
# the pairings in which the router lost to the best fixed schedule at 1e-8 with the unit of one corrector call (128^2):
# the granularity ablation runs them first
LOSING="Poisson:gs,jacobi_0.67,ssor ConvDiff:jacobi_0.67,ssor AnisoDiff:jacobi_0.67,linegs VarCoeff:jacobi_0.67"
ONLY_N=128 SKIP_AUX=1 ./run_final.sh \
  && ONLY_N=128 SKIP_AUX=1 ./run_varcoeff.sh \
  && ./run_retime128.sh \
  && note "128^2 core complete: pairwise cells and ensembles of all four equations" \
  && GRAN_ONLY=$LOSING ./run_granularity.sh \
  && note "granularity ablation (unit = corrector / 4) done for the eight pairings the router lost at 1e-8" \
  && ONLY_N=256 SKIP_AUX=1 ONLY_EQ=Poisson ./run_final.sh \
  && note "256^2 Poisson cells done (brought forward: the larger grid is the other candidate setting against the best fixed schedule)" \
  && ONLY_N=256 SKIP_AUX=1 ONLY_EQ=ConvDiff ./run_final.sh \
  && note "256^2 ConvDiff cells done (brought forward)" \
  && ONLY_N=128 ./run_final.sh \
  && ONLY_N=128 ./run_varcoeff.sh \
  && ./run_granularity.sh \
  && note "128^2 phase complete (auxiliary checks and granularity ablation included); larger grids next" \
  && ./run_final.sh \
  && ./run_varcoeff.sh
rc=$?
echo "$(date '+%F %T') run_all finished with status $rc" >> logs/final_progress.log
