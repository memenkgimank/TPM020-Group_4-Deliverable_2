"""
run_sample_tuning.py
--------------------
Runs Step 4 (sample tuning) entirely from a .py script.

What it does
- Calls tune_with_threshold.py with the dataset + grids you set below
- Writes: tuning_results_tech_and_epistemic.csv
- Prints the top rows so you can pick a config for the full run
"""
import subprocess
import sys
from pathlib import Path
import shlex

# ========== CONFIG ==========
CSV_PATH = "amazon_data/Amazon_IoT_product_reviews.csv"
TEXT_COL = "Review_body"
LABEL_COL = "Rating"
N_SAMPLES = 500

MAX_LENGTHS = [128, 256, 384]
CHUNK_LONG_OPTS = [0, 1]
AGG_OPTS = ["mean", "max"]
BATCH_SIZES = [16, 32]

DECISION_MODES = ["argmax", "top_p", "one_vs_rest"]
TOP_P_THRESHOLDS = [0.0, 0.6, 0.7]
OVR_THRESHOLDS = [0.6, 0.7]
COLLAPSE_NEUTRAL_OPTS = [0, 1]
# ============================

def main():
    here = Path(__file__).parent.resolve()
    tuner = here / "tune_with_threshold.py"  # <-- singular filename
    if not tuner.exists():
        raise FileNotFoundError(f"Could not find tune_with_threshold.py at: {tuner}")

    cmd = [
        sys.executable, str(tuner),
        "--csv", CSV_PATH,
        "--text_col", TEXT_COL,
        "--label_col", LABEL_COL,
        "--n_samples", str(N_SAMPLES),
        "--max_lengths", *map(str, MAX_LENGTHS),
        "--chunk_long_opts", *map(str, CHUNK_LONG_OPTS),
        "--agg_opts", *AGG_OPTS,
        "--batch_sizes", *map(str, BATCH_SIZES),
        "--decision_modes", *DECISION_MODES,
        "--top_p_thresholds", *map(str, TOP_P_THRESHOLDS),
        "--ovr_thresholds", *map(str, OVR_THRESHOLDS),
        "--collapse_neutral_opts", *map(str, COLLAPSE_NEUTRAL_OPTS),
    ]

    print(">>> running tuner:\n", " ".join(shlex.quote(c) for c in cmd), "\n")
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("\n✅ Done. Look for 'tuning_results_tech_and_epistemic.csv' in this folder.")

if __name__ == "__main__":
    main()
