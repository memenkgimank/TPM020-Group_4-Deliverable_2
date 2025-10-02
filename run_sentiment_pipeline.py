"""
run_sentiment_pipeline.py
=========================
Purpose
-------
Single entrypoint that lets you run either:
- TUNING (sample grid search of technical + epistemic settings), or
- FULL RUN (apply chosen settings to the entire CSV).

It simply dispatches to:
- tune_with_thresholds.py
- run_full_inference_with_thresholds.py

Subcommands
-----------
1) tune  : Run sample tuning and write tuning_results_tech_and_epistemic.csv
2) full  : Run full inference and write predictions.parquet

Common inputs
-------------
--csv <path>       : input CSV
--text_col <name>  : text column (e.g., Review_body)

Tuning options (subset)
-----------------------
--label_col <name>         : e.g., Rating (1–5 mapped to 3-class)
--n_samples <int>          : sample size (default 3000)
--max_lengths 128 256 384
--chunk_long_opts 0 1
--agg_opts mean max
--batch_sizes 16 32
--decision_modes argmax top_p one_vs_rest
--top_p_thresholds 0.6 0.7
--ovr_thresholds 0.6 0.7
--collapse_neutral_opts 0 1

Full-run options (subset)
-------------------------
--out predictions.parquet
--max_length 384
--chunk_long 1
--agg mean
--batch_size 32
--fp16 1
--decision_mode top_p
--top_p_threshold 0.7
--ovr_threshold 0.6
--collapse_neutral 0

Examples
--------
# 1) Tuning on a sample
python run_sentiment_pipeline.py tune \
  --csv Amazon_IoT_product_reviews.csv \
  --text_col Review_body \
  --label_col Rating \
  --n_samples 3000

# 2) Full run with chosen settings
python run_sentiment_pipeline.py full \
  --csv Amazon_IoT_product_reviews.csv \
  --text_col Review_body \
  --out predictions.parquet \
  --max_length 384 --chunk_long 1 --agg mean --batch_size 32 --fp16 1 \
  --decision_mode top_p --top_p_threshold 0.7 --collapse_neutral 0

Notes
-----
- Keep this file in the same folder as the other scripts.
- It prints the exact command it executes under the hood for transparency.
"""

"""
run_sentiment_pipeline.py
=========================
Single entrypoint that runs either:
- TUNING (sample grid search of technical + epistemic settings), or
- FULL RUN (apply chosen settings to the entire CSV).
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

def run(cmd_list):
    print(">>>", " ".join(shlex.quote(c) for c in cmd_list))
    p = subprocess.Popen(cmd_list)
    p.wait()
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def build_and_run_tune(args):
    script = str(HERE / "tune_with_threshold.py")  # <-- singular
    cmd = [sys.executable, script,
           "--csv", args.csv,
           "--text_col", args.text_col,
           "--label_col", args.label_col,
           "--n_samples", str(args.n_samples)]
    if args.max_lengths: cmd += ["--max_lengths", *map(str, args.max_lengths)]
    if args.chunk_long_opts is not None: cmd += ["--chunk_long_opts", *map(str, args.chunk_long_opts)]
    if args.agg_opts: cmd += ["--agg_opts", *args.agg_opts]
    if args.batch_sizes: cmd += ["--batch_sizes", *map(str, args.batch_sizes)]
    if args.decision_modes: cmd += ["--decision_modes", *args.decision_modes]
    if args.top_p_thresholds: cmd += ["--top_p_thresholds", *map(str, args.top_p_thresholds)]
    if args.ovr_thresholds: cmd += ["--ovr_thresholds", *map(str, args.ovr_thresholds)]
    if args.collapse_neutral_opts is not None: cmd += ["--collapse_neutral_opts", *map(str, args.collapse_neutral_opts)]
    run(cmd)

def build_and_run_full(args):
    script = str(HERE / "run_full_inference_with_threshold.py")  # <-- singular
    cmd = [sys.executable, script,
           "--csv", args.csv,
           "--text_col", args.text_col,
           "--out", args.out,
           "--max_length", str(args.max_length),
           "--chunk_long", str(args.chunk_long),
           "--agg", args.agg,
           "--batch_size", str(args.batch_size),
           "--fp16", str(args.fp16),
           "--chunksize", str(args.chunksize),
           "--decision_mode", args.decision_mode,
           "--top_p_threshold", str(args.top_p_threshold),
           "--ovr_threshold", str(args.ovr_threshold),
           "--collapse_neutral", str(args.collapse_neutral)]
    run(cmd)

def main():
    ap = argparse.ArgumentParser(prog="run_sentiment_pipeline.py")
    sub = ap.add_subparsers(dest="mode", required=True)

    # TUNE subcommand
    tune = sub.add_parser("tune", help="Run technical + epistemic tuning on a sample")
    tune.add_argument("--csv", default="Amazon_IoT_product_reviews.csv")
    tune.add_argument("--text_col", default="Review_body")
    tune.add_argument("--label_col", default="Rating")
    tune.add_argument("--n_samples", type=int, default=3000)

    tune.add_argument("--max_lengths", type=int, nargs="+", default=[128, 256, 384])
    tune.add_argument("--chunk_long_opts", type=int, nargs="+", default=[0, 1])
    tune.add_argument("--agg_opts", nargs="+", default=["mean", "max"])
    tune.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32])

    tune.add_argument("--decision_modes", nargs="+", default=["argmax", "top_p", "one_vs_rest"])
    tune.add_argument("--top_p_thresholds", type=float, nargs="+", default=[0.0, 0.6, 0.7])
    tune.add_argument("--ovr_thresholds", type=float, nargs="+", default=[0.6, 0.7])
    tune.add_argument("--collapse_neutral_opts", type=int, nargs="+", default=[0, 1])

    tune.set_defaults(func=build_and_run_tune)

    # FULL subcommand
    full = sub.add_parser("full", help="Run full-dataset inference with chosen settings")
    full.add_argument("--csv", default="Amazon_IoT_product_reviews.csv")
    full.add_argument("--text_col", default="Review_body")
    full.add_argument("--out", default="predictions.parquet")

    full.add_argument("--max_length", type=int, default=256)
    full.add_argument("--chunk_long", type=int, choices=[0, 1], default=1)
    full.add_argument("--agg", choices=["mean", "max"], default="mean")
    full.add_argument("--batch_size", type=int, default=32)
    full.add_argument("--fp16", type=int, choices=[0, 1], default=1)
    full.add_argument("--chunksize", type=int, default=20000)

    full.add_argument("--decision_mode", choices=["argmax", "top_p", "one_vs_rest"], default="top_p")
    full.add_argument("--top_p_threshold", type=float, default=0.7)
    full.add_argument("--ovr_threshold", type=float, default=0.6)
    full.add_argument("--collapse_neutral", type=int, choices=[0, 1], default=0)

    full.set_defaults(func=build_and_run_full)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
