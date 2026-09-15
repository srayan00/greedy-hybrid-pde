#!/bin/zsh
# Re-timing pass for the 128^2 pairwise cells written by an earlier revision of the driver. A cell is re-timed when
# its policy rows lack the stored operation sequences, its baseline rows lack the Krylov validity flag, or its drift
# records lack the reference measurements (in the revision-2 run: the Poisson and convection-diffusion cells). Every
# instance is timed again with the final driver (--retime_all), using the stored routers and cost calibration
# (bench.py refuses if either differs), the drift guard waiting up to 5 min per instance; the result file keeps its
# original provenance and adds the re-timing provenance. Cells written by the final driver are left untouched, and a
# re-timed cell no longer matches the criterion, so the pass resumes after an interruption.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-/Users/yash/miniconda3/envs/ansatz/bin/python}
POL=classical,hints2,hints5,hints10,hints15,hints25,hints50,phints5,phints10,phints15,phints25,phints50,oneshot,decay0.9,decay0.95,decay0.98,greedy,oracle,router
stage() { echo "$(date '+%F %T') $1" >> logs/final_progress.log; }
fail() { echo "$(date '+%F %T') FAILED: $1" >> logs/final_progress.log; echo FAILED > logs/retime.failed; exit 1; }
run() { "$@" || fail "(exit $?) $*"; }
rm -f logs/retime.failed
# exit status 0: the cell predates the final driver; 1: it does not; anything else: the file cannot be read
needs_retime() {
  $PY - "$1" <<'EOF'
import json, sys
try:
    g = next(iter(json.load(open(sys.argv[1]))["groups"].values()))
    P = g["policies"]
    old = (any(p.startswith("base:") and rows and "valid" not in rows[0] for p, rows in P.items())
           or any(not p.startswith("base:") and rows and "op_rle" not in rows[0] for p, rows in P.items())
           or (bool(g.get("drift")) and "ref_us" not in g["drift"][0]))
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)
sys.exit(0 if old else 1)
EOF
}
new=0
for EQ in Poisson ConvDiff AnisoDiff; do for S in $(cat config/solvers_$EQ | tr , ' '); do
  f=results/${EQ}_128_${S}.json
  [ -f $f ] || fail "$f is missing before the re-timing pass"
  needs_retime $f; rc=$?
  [ $rc = 1 ] && continue
  [ $rc = 0 ] || fail "cannot read $f"
  run $PY bench.py --equation $EQ --N 128 --solvers $S --n_test 64 --max_ops 60000 --policies $POL --seed 73 --timed_reps 3 --retime_only --retime_all > logs/final_retime_${EQ}_128_$S.log 2>&1
  new=1
done; done
if [ $new = 1 ]; then stage "re-timing pass done (128^2 cells of earlier driver revisions)"; fi
exit 0
