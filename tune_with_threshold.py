import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sentiment_infer import TwitterRobertaSentiment, InferenceConfig, LABELS

# ---------- Helpers ----------

def map_stars_to_3class(series: pd.Series) -> Optional[np.ndarray]:
    """
    Map 1-5 star ratings to 3-class sentiments: 1-2->neg(0), 3->neu(1), 4-5->pos(2).
    Return None if series cannot be parsed as star ratings or 0/1/2 labels.
    """
    s = series.dropna()
    # Already 0/1/2?
    uniq = set(s.unique())
    if uniq.issubset({0,1,2}):
        return series.values.astype(int)

    # Try to coerce to integers 1-5
    try:
        ints = series.astype(float).round().astype(int)
    except Exception:
        return None

    if set(ints.dropna().unique()).issubset({1,2,3,4,5}):
        mapped = ints.map(lambda x: 0 if x <= 2 else (1 if x == 3 else 2)).values
        return mapped
    return None

def apply_decision_rules(
    probs: np.ndarray,
    decision_mode: str = "argmax",
    top_p_threshold: float = 0.0,
    one_vs_rest_thresholds: Dict[str, float] = None,
    add_uncertain: bool = False,
    collapse_neutral: bool = False,
):
    """
    decision_mode:
      - "argmax": pick highest-prob class.
      - "top_p": if max(prob) >= top_p_threshold -> argmax else label = 3 (uncertain).
      - "one_vs_rest": if any class prob >= its threshold pick the highest that passes;
                       otherwise label = 3 (uncertain).
    add_uncertain: if True, label 3 is used for uncertain; otherwise we still output 3 but caller may drop.
    collapse_neutral: if True, map neutral (1) into neg(0) if neg>pos else pos(2) *after* deciding.
    Returns: y_pred (np.ndarray) of shape (n,) with labels in {0,1,2,3?}
    """
    n = probs.shape[0]
    y = np.zeros(n, dtype=int)

    if decision_mode == "argmax":
        y = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        if add_uncertain and top_p_threshold > 0.0:
            y = np.where(conf >= top_p_threshold, y, 3)

    elif decision_mode == "top_p":
        conf = probs.max(axis=1)
        y = probs.argmax(axis=1)
        y = np.where(conf >= top_p_threshold, y, 3)

    elif decision_mode == "one_vs_rest":
        if one_vs_rest_thresholds is None:
            one_vs_rest_thresholds = {"negative": 0.6, "neutral": 0.6, "positive": 0.6}
        y = np.full(n, 3, dtype=int)  # default uncertain
        for i in range(n):
            passed = []
            for idx, name in enumerate(LABELS):
                if probs[i, idx] >= one_vs_rest_thresholds.get(name, 0.6):
                    passed.append(idx)
            if passed:
                best_idx = passed[np.argmax(probs[i, passed])]
                y[i] = best_idx
    else:
        raise ValueError(f"Unknown decision_mode: {decision_mode}")

    if collapse_neutral:
        pos_ge_neg = probs[:, 2] >= probs[:, 0]
        y = np.where(y == 1, np.where(pos_ge_neg, 2, 0), y)

    return y

def eval_with_optional_uncertain(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute metrics:
       - Filtered 3-class (drop uncertain)
       - Report kept_fraction + distribution if uncertain present
    """
    results = {}
    has_uncertain = np.any(y_pred == 3)

    mask = y_pred != 3
    if mask.sum() > 0:
        acc = accuracy_score(y_true[mask], y_pred[mask])
        macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
        results.update({
            "acc_3class_filtered": acc,
            "macro_f1_3class_filtered": macro_f1,
            "kept_fraction": float(mask.mean()),
        })
    else:
        results.update({
            "acc_3class_filtered": np.nan,
            "macro_f1_3class_filtered": np.nan,
            "kept_fraction": 0.0,
        })

    if has_uncertain:
        unique, counts = np.unique(y_pred, return_counts=True)
        results["pred_label_distribution"] = {int(k): int(v) for k,v in zip(unique, counts)}
    return results

# ---------- Main tuning ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text_col", default="Review_body")
    ap.add_argument("--label_col", default="Rating")
    ap.add_argument("--n_samples", type=int, default=2000)
    # Technical grid
    ap.add_argument("--max_lengths", type=int, nargs="+", default=[128, 256, 384])
    ap.add_argument("--chunk_long_opts", type=int, nargs="+", default=[0,1])  # 0/1
    ap.add_argument("--agg_opts", nargs="+", default=["mean", "max"])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32])
    # Epistemic grid
    ap.add_argument("--decision_modes", nargs="+", default=["argmax", "top_p", "one_vs_rest"])
    ap.add_argument("--top_p_thresholds", type=float, nargs="+", default=[0.0, 0.6, 0.7])
    ap.add_argument("--ovr_thresholds", type=float, nargs="+", default=[0.6, 0.7])
    ap.add_argument("--collapse_neutral_opts", type=int, nargs="+", default=[0,1])  # 0/1
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col])
    if len(df) > args.n_samples:
        df = df.sample(args.n_samples, random_state=13)

    texts = df[args.text_col].astype(str).tolist()
    y_true = None
    if args.label_col in df.columns:
        y_true = map_stars_to_3class(df[args.label_col])

    results: List[Dict[str, Any]] = []

    for max_length in args.max_lengths:
        for chunk_flag in args.chunk_long_opts:
            for agg in (args.agg_opts if chunk_flag else ["mean"]):
                for bs in args.batch_sizes:
                    cfg = InferenceConfig(
                        max_length=max_length,
                        chunk_long=bool(chunk_flag),
                        agg=agg,
                        batch_size=bs,
                    )
                    clf = TwitterRobertaSentiment(cfg)
                    probs = clf.predict_proba(texts)

                    for mode in args.decision_modes:
                        if mode == "argmax":
                            top_p_list = [0.0]
                            ovr_list = [None]
                        elif mode == "top_p":
                            top_p_list = args.top_p_thresholds
                            ovr_list = [None]
                        else:  # one_vs_rest
                            top_p_list = [0.0]
                            ovr_list = args.ovr_thresholds

                        for top_p in top_p_list:
                            for ovr in ovr_list:
                                for collapse_neutral in args.collapse_neutral_opts:
                                    if mode == "one_vs_rest":
                                        thr_dict = {"negative": ovr, "neutral": ovr, "positive": ovr}
                                    else:
                                        thr_dict = None

                                    y_pred = apply_decision_rules(
                                        probs,
                                        decision_mode=mode,
                                        top_p_threshold=top_p,
                                        one_vs_rest_thresholds=thr_dict,
                                        add_uncertain=True,
                                        collapse_neutral=bool(collapse_neutral),
                                    )

                                    row = dict(
                                        max_length=max_length,
                                        chunk_long=int(chunk_flag),
                                        agg=agg,
                                        batch_size=bs,
                                        decision_mode=mode,
                                        top_p_threshold=top_p,
                                        ovr_threshold=ovr if ovr is not None else np.nan,
                                        collapse_neutral=int(collapse_neutral),
                                    )

                                    if y_true is not None:
                                        metrics = eval_with_optional_uncertain(y_true, y_pred)
                                        row.update(metrics)
                                    else:
                                        unique, counts = np.unique(y_pred, return_counts=True)
                                        dist = {int(k): float(counts[idx]/len(y_pred)) for idx, k in enumerate(unique)}
                                        row.update(pred_label_distribution=dist)

                                    results.append(row)

    out = pd.DataFrame(results)

    # Sort if we have labels
    sort_cols = []
    if "macro_f1_3class_filtered" in out.columns:
        sort_cols = ["macro_f1_3class_filtered", "acc_3class_filtered", "kept_fraction"]

    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=[False, False, False])

    out_path = "tuning_results_tech_and_epistemic.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if sort_cols:
        print("Top 5 configs:\n", out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
