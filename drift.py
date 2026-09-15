"""Drift-guard bookkeeping shared by bench.py and run_retime128.sh (numpy only, no solver imports)."""
import numpy as np


def final_reference_us(drift):
    """The cell's final reference: the 10th percentile of the reference readings (microseconds) under
    which its instances were timed, or None for records without readings (earlier driver revisions)."""
    refs = [dr["ref_us"] for dr in drift if dr.get("ref_us")]
    return float(np.percentile(refs, 10)) if refs else None


def slow_instances(drift, tol):
    """Instances timed while the reference was more than `tol` slower than the session reference at the
    check (the guard gave up) or than the cell's final reference (a session whose early reference, and
    hence the guard's comparison, was itself slow)."""
    ref = final_reference_us(drift)
    return [dr["instance"] for dr in drift
            if dr["ratio"] > 1.0 + tol or (ref is not None and dr.get("ref_us") and dr["ref_us"] / ref > 1.0 + tol)]
