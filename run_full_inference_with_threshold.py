"""
run_full_inference_with_thresholds.py
=====================================
Purpose
-------
Run sentiment inference over the *entire* dataset using a single configuration
you selected from tuning. Streams the CSV in chunks to control memory.

What it does
------------
- Reads the input CSV in chunks (e.g., 20k rows at a time).
- Applies your technical settings: max_length, chunk_long, agg, batch_size, fp16.
- Applies your epistemic rule:
  * argmax
  * top_p (top class prob must exceed threshold; else 'uncertain')
  * one_vs_rest (per-class thresholds; else 'uncertain')
  * optional neutral collapsing
- Writes output as a Parquet file with added columns:
  pred_neg, pred_neu, pred_pos, pred_label_id, pred_label

Inputs (CLI)
------------
--csv <path>                : full dataset CSV path
--text_col <name>           : text column name (e.g., Review_body)
--out <path>                : output Parquet path (e.g., predictions.parquet)
--max_length <int>          : 128/256/384 (longer = more context, slower)
--chunk_long {0,1}          : chunk long reviews (recommended 1)
--agg {mean,max}            : chunk aggregation
--batch_size <int>          : GPU/CPU memory trade-off
--fp16 {0,1}                : 1 for GPU half precision; 0 for CPU
--chunksize <int>           : CSV streaming chunk size (default 20000)
--decision_mode <mode>      : argmax | top_p | one_vs_rest
--top_p_threshold <float>   : used with top_p (e.g., 0.7)
--ovr_threshold <float>     : used with one_vs_rest (e.g., 0.6)
--collapse_neutral {0,1}    : merge neutral into pos/neg after decision

Outputs
-------
- Parquet with original columns + predictions:
  * pred_neg, pred_neu, pred_pos (probabilities)
  * pred_label_id (0=neg, 1=neu, 2=pos, 3=uncertain)
  * pred_label ("negative", "neutral", "positive", "uncertain")

Example
-------
python run_full_inference_with_thresholds.py \
  --csv Amazon_IoT_product_reviews.csv \
  --text_col Review_body \
  --out predictions.parquet \
  --max_length 384 --chunk_long 1 --agg mean --batch_size 32 --fp16 1 \
  --decision_mode top_p --top_p_threshold 0.7 --collapse_neutral 0

Notes
-----
- For binary-only outputs, set --collapse_neutral 1.
- If you hit OOM, try smaller --batch_size or lower --max_length.
"""

import argparse
import numpy as np
import pandas as pd
from sentiment_infer import TwitterRobertaSentiment, InferenceConfig, LABELS
# reuse epistemic rule function from the tuner
# NEW (matches your tuner filename)
from tune_with_threshold import apply_decision_rules


def batched(seq, n):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text_col", default="Review_body")
    ap.add_argument("--out", default="predictions.parquet")
    # Technical
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--chunk_long", type=int, default=1)  # 1/0
    ap.add_argument("--agg", choices=["mean","max"], default="mean")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=20000)
    # Epistemic
    ap.add_argument("--decision_mode", choices=["argmax","top_p","one_vs_rest"], default="top_p")
    ap.add_argument("--top_p_threshold", type=float, default=0.7)
    ap.add_argument("--ovr_threshold", type=float, default=0.6)
    ap.add_argument("--collapse_neutral", type=int, default=0)
    args = ap.parse_args()

    cfg = InferenceConfig(
        max_length=args.max_length,
        chunk_long=bool(args.chunk_long),
        agg=args.agg,
        batch_size=args.batch_size,
        fp16=bool(args.fp16),
    )
    clf = TwitterRobertaSentiment(cfg)

    first = True
    total = 0

    for df in pd.read_csv(args.csv, chunksize=args.chunksize):
        texts = df[args.text_col].fillna("").astype(str).tolist()

        # micro-batch to stabilize memory on big datasets
        probs_list = []
        for tb in batched(texts, 5000):
            probs = clf.predict_proba(tb)
            probs_list.append(probs)
        probs_all = np.vstack(probs_list)

        # apply epistemic rule
        if args.decision_mode == "one_vs_rest":
            thr_dict = {"negative": args.ovr_threshold, "neutral": args.ovr_threshold, "positive": args.ovr_threshold}
        else:
            thr_dict = None

        y_pred = apply_decision_rules(
            probs_all,
            decision_mode=args.decision_mode,
            top_p_threshold=args.top_p_threshold,
            one_vs_rest_thresholds=thr_dict,
            add_uncertain=True,
            collapse_neutral=bool(args.collapse_neutral),
        )

        pred_labels = np.where(y_pred == 3, "uncertain", np.array(LABELS, dtype=object)[y_pred])
        out_df = pd.DataFrame({
            "pred_neg": probs_all[:,0],
            "pred_neu": probs_all[:,1],
            "pred_pos": probs_all[:,2],
            "pred_label_id": y_pred,
            "pred_label": pred_labels,
        })

        merged = pd.concat([df.reset_index(drop=True), out_df], axis=1)

        merged.to_parquet(
            args.out,
            index=False,
            engine="pyarrow",
            compression="zstd",
            append=not first
        )
        first = False
        total += len(df)

    print(f"Done. Wrote {total} rows -> {args.out}")

if __name__ == "__main__":
    main()
