#!/bin/zsh
# Master chain, ordered so that every 128^2 result (all four equations) is complete before any larger grid:
#   1. 128^2 pairwise cells (with same-session baselines) and ensembles of Poisson, ConvDiff, AnisoDiff
#   2. 128^2 variable-coefficient diffusion: corrector, pairwise cells, nested ensembles
#   3. re-timing pass of the 128^2 cells written by an earlier driver revision
#   4. 128^2 usage traces, seed trials and assumption / screening / theorem checks (all four equations)
#   5. decision-granularity ablation (128^2 only)
#   6. 256^2 and 512^2 cells with their ensembles and checks, overheads, discretisation study
# Every script skips steps whose output already exists, so the chain resumes after an interruption. Launch detached:
#   nohup ./run_all.sh > logs/run_all.out 2>&1 &
cd "$(dirname "$0")"
note() { echo "$(date '+%F %T') ===== $1 =====" >> logs/final_progress.log; }
ONLY_N=128 SKIP_AUX=1 ./run_final.sh \
  && ONLY_N=128 SKIP_AUX=1 ./run_varcoeff.sh \
  && ./run_retime128.sh \
  && note "128^2 core complete: pairwise cells and ensembles of all four equations" \
  && ONLY_N=128 ./run_final.sh \
  && ONLY_N=128 ./run_varcoeff.sh \
  && ./run_granularity.sh \
  && note "128^2 phase complete (auxiliary checks and granularity ablation included); larger grids next" \
  && ./run_final.sh \
  && ./run_varcoeff.sh
echo "$(date '+%F %T') run_all finished with status $?" >> logs/final_progress.log
