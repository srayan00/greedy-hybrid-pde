#!/bin/zsh
# Re-timing pass for the 128^2 pairwise cells, decided from the stored data:
#  (1) cells written by an earlier revision of the driver (policy rows without stored operation sequences, baseline rows
#      without the Krylov validity flag, or drift records without reference readings; in the revision-2 run the Poisson
#      and convection-diffusion cells) are re-timed completely with the final driver (--retime_all);
#  (2) cells of the current format that contain instances timed while the reference operation was more than 15% slower
#      than the session reference at the check or than the cell's final reference (drift.py) have those instances
#      re-timed (--retime_only).
# Both use the stored routers and cost calibration (bench.py refuses if either differs), anchor the drift guard to the
# stored reference readings, and keep the original provenance next to the re-timing provenance. A re-timed cell no
# longer matches either criterion unless an instance stayed slow through the 5-min wait, so the pass can be resumed.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
fail() { echo "$(date '+%F %T') FAILED: $1" >> logs/final_progress.log; echo FAILED > logs/retime.failed; exit 1; }
run() { "$@" || fail "(exit $?) $*"; }
rm -f logs/retime.failed
# exit status 0: re-time every instance; 3: re-time the slow instances; 1: nothing to do; anything else: unreadable
needs_retime() {
  $PY - "$1" <<'EOF'
import json, sys
sys.path.insert(0, ".")
try:
    from drift import slow_instances
    g = next(iter(json.load(open(sys.argv[1]))["groups"].values()))
    P = g["policies"]
    old = (any(p.startswith("base:") and rows and "valid" not in rows[0] for p, rows in P.items())
           or any(not p.startswith("base:") and rows and "op_rle" not in rows[0] for p, rows in P.items())
           or (bool(g.get("drift")) and "ref_us" not in g["drift"][0]))
    code = 0 if old else (3 if slow_instances(g.get("drift", []), 0.15) else 1)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)
sys.exit(code)
EOF
}
new=0
for EQ in Poisson ConvDiff AnisoDiff; do for S in $(cat config/solvers_$EQ | tr , ' '); do
  f=results/${EQ}_128_${S}.json
  [ -f $f ] || fail "$f is missing before the re-timing pass"
  needs_retime $f; rc=$?
  case $rc in
    1) continue ;;
    0) MODE="--retime_only --retime_all" ;;
    3) MODE="--retime_only" ;;
    *) fail "cannot read $f" ;;
  esac
  run $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --policies $POL --seed 73 --timed_reps 3 ${=MODE} > logs/final_retime_${EQ}_128_$S.log 2>&1
  new=1
done; done
if [ $new = 1 ]; then stage "re-timing pass done (128^2 cells)"; fi
exit 0
