#!/bin/zsh
# Master chain: main study -> variable-coefficient diffusion -> granularity ablation. Launch detached:
#   nohup ./run_all.sh > logs/run_all.out 2>&1 &
cd "$(dirname "$0")"
./run_final.sh && ./run_varcoeff.sh && ./run_granularity.sh
echo "$(date '+%F %T') run_all finished with status $?" >> logs/final_progress.log
