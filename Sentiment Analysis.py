import pandas as pd
import numpy as np
import ssl, certifi, nltk, os
# --- SSL + NLTK bootstrap (put at the very top, before any nltk usage) ---
import os, ssl, certifi, nltk

# Use certifi's CA bundle for HTTPS downloads (e.g., NLTK)
os.environ["SSL_CERT_FILE"] = certifi.where()

def _sslcontext(*args, **kwargs):
    # return a fresh SSLContext that trusts certifi CAs
    return ssl.create_default_context(cafile=certifi.where())

# urllib (used by NLTK downloader) consults this factory
ssl._create_default_https_context = _sslcontext

# Tell NLTK where to store/find data, then ensure downloads
NLTK_HOME = os.path.expanduser("~/nltk_data")
os.makedirs(NLTK_HOME, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_HOME

for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_HOME, quiet=False)

from nltk.tokenize import sent_tokenize
# --- end bootstrap ---

from transformers import pipeline

# 1) Zero-shot classifier to detect privacy/security-relevant sentences
zshot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2) Your sentiment model
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


PRIV_LABELS = ["privacy/security-related", "not privacy/security-related"]
HYP = "This sentence is {}."
REL_THRESH = 0.70  # tune between 0.6–0.8

def extract_privsec_sentences(text: str):
    sents = [s.strip() for s in sent_tokenize(str(text)) if s.strip()]
    if not sents:
        return []
    res = zshot(sents, PRIV_LABELS, hypothesis_template=HYP, multi_label=False)
    # pipeline returns dict or list of dicts depending on input length
    if isinstance(res, dict): res = [res]
    keep = []
    for sent, out in zip(sents, res):
        lbls = out["labels"]; scs = out["scores"]
        score_map = dict(zip(lbls, scs))
        if score_map.get("privacy/security-related", 0.0) >= REL_THRESH:
            keep.append(sent)
    return keep



LABEL_MAP = {"negative": -1, "neutral": 0, "positive": 1}

def sentiment_on_privsec_sentences(text: str):
    sents = extract_privsec_sentences(text)
    if not sents:
        return {"has_privsec": False, "agg_label": None, "agg_score": None, "sentiments": []}
    outs = sentiment(sents)
    # normalize to (-1,0,1) * score so “most negative” = min
    scored = []
    for s, o in zip(sents, outs):
        lbl = o["label"].lower()
        sc = float(o["score"])
        scored.append({"sentence": s, "label": lbl, "score": sc, "signed": LABEL_MAP[lbl]*sc})
    # choose the most negative sentence
    worst = min(scored, key=lambda d: d["signed"])
    return {"has_privsec": True, "agg_label": worst["label"], "agg_score": worst["score"], "sentiments": scored}



# Load your reviews
df = pd.read_excel("amazon_data/test.xlsx").dropna(subset=["Review_body", "Country"])

# Run sentence-level extraction + sentiment
results = df["Review_body"].apply(sentiment_on_privsec_sentences)

df["has_privsec"] = results.apply(lambda x: x["has_privsec"])
df["privsec_sentiments"] = results.apply(lambda x: x["sentiments"])
df["privsec_agg_label"] = results.apply(lambda x: x["agg_label"])
df["privsec_agg_score"] = results.apply(lambda x: x["agg_score"])

# Keep only reviews with privacy/security content and negative agg label
neg_df = df[(df["has_privsec"]) & (df["privsec_agg_label"]=="negative")].copy()

# For each country, pick the 10–15 most negative (lowest signed score); tie-break by helpful votes if present
def worst_key(row):
    # use min of signed scores among sentences to rank severity
    signed_vals = [d["signed"] for d in row["privsec_sentiments"]]
    m = min(signed_vals) if signed_vals else 0
    hv = row.get("Number_helpful", 0) if "Number_helpful" in row else 0
    return (m, -hv)

top_bad_by_country = (
    neg_df
    .sort_values(by=["Country"], kind="stable", key=None)
    .groupby("Country", group_keys=False)
    .apply(lambda g: g.sort_values(by=g.index.map(lambda i: worst_key(g.loc[i]))).head(15))
)


ISSUE_LABELS = [
    "Authentication and passwords",
    "Account/registration and consent",
    "App permissions and data sharing",
    "Firmware updates and patching",
    "Connectivity, pairing, or setup (WPS, QR, Wi-Fi)",
    "Encryption and standards (WPA/WPA3, TLS/SSL)",
    "Remote access, cloud storage, or camera feed exposure",
    "Usability friction with security features (2FA, CAPTCHAs)",
    "Customer support and policy regarding security"
]
ISSUE_HYP = "This sentence is primarily about {}."

def issue_tag_sentence(sentence: str):
    out = zshot(sentence, ISSUE_LABELS, hypothesis_template=ISSUE_HYP, multi_label=False)
    return {"label": out["labels"][0], "score": float(out["scores"][0])}

def tag_top_bad_issues(df_top: pd.DataFrame, conf_thresh=0.50):
    rows = []
    for idx, r in df_top.iterrows():
        for s in r["privsec_sentiments"]:
            if s["label"]=="negative":  # focus on the negative sentences
                tag = issue_tag_sentence(s["sentence"])
                if tag["score"] >= conf_thresh:
                    rows.append({
                        "Country": r["Country"],
                        "Index": r.get("Index", None),
                        "sentence": s["sentence"],
                        "issue_label": tag["label"],
                        "issue_score": tag["score"]
                    })
    return pd.DataFrame(rows)

issues_df = tag_top_bad_issues(top_bad_by_country, conf_thresh=0.55)


# Top issue categories per country
issues_by_country = (
    issues_df
    .groupby(["Country","issue_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["Country","count"], ascending=[True,False])
)

# Overall top issues
overall_issues = (
    issues_df["issue_label"]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index":"issue_label"})
)

print(issues_by_country.head(20))
print(overall_issues.head(10))
