"""Drift-guard bookkeeping shared by bench.py and run_retime128.sh (numpy only, no solver imports)."""
import json
import os

import numpy as np

INDEX = "drift_refs.json"   # final reference readings of the saved cells, kept next to the results


def final_reference_us(drift):
    """The cell's final reference: the 10th percentile of the reference readings (microseconds) under
    which its instances were timed, or None for records without readings (earlier driver revisions)."""
    refs = [dr["ref_us"] for dr in drift if dr.get("ref_us")]
    return float(np.percentile(refs, 10)) if refs else None


def load_index(out_dir):
    """{cell file: {equation, N, ref_final_us}} of the saved cells (empty if there is no index yet)."""
    try:
        with open(os.path.join(out_dir, INDEX)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def record_final_reference(out_dir, cell, equation, N, ref_us):
    """Stores a saved cell's final reference in the index (written atomically)."""
    idx = load_index(out_dir)
    idx[cell] = {"equation": equation, "N": int(N), "ref_final_us": ref_us}
    tmp = os.path.join(out_dir, INDEX + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(idx, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, INDEX))


def cross_session_anchor_us(out_dir, equation, N, exclude=None, min_cells=3):
    """The median final reference of the other saved cells of the same equation and grid, or None with fewer
    than `min_cells` of them (a session that ran slow throughout has a slow final reference of its own)."""
    vals = [v["ref_final_us"] for k, v in load_index(out_dir).items()
            if k != exclude and v.get("equation") == equation and v.get("N") == int(N) and v.get("ref_final_us")]
    return float(np.median(vals)) if len(vals) >= min_cells else None


def slow_instances(drift, tol, anchor_us=None):
    """Instances timed while the reference was more than `tol` slower than the session reference at the
    check (the guard gave up), than the cell's final reference (a session whose early reference was itself
    slow) or than the cross-session anchor (a session that was slow throughout)."""
    ref = final_reference_us(drift)
    if anchor_us is not None:
        ref = anchor_us if ref is None else min(ref, anchor_us)
    return [dr["instance"] for dr in drift
            if dr["ratio"] > 1.0 + tol or (ref is not None and dr.get("ref_us") and dr["ref_us"] / ref > 1.0 + tol)]
