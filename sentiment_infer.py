"""
sentiment_infer.py
==================
Purpose
-------
Core inference engine that loads the pretrained `cardiffnlp/twitter-roberta-base-sentiment-latest`
model and runs sentiment analysis on text. No training is performed.

What it does
------------
- Converts input texts to probabilities for 3 classes: [negative, neutral, positive].
- Handles long reviews via chunking + aggregation (mean/max).
- Exposes technical knobs: max_length, chunk_long, stride, agg, batch_size, fp16.

Key classes
-----------
- InferenceConfig: configuration dataclass for inference behavior.
- TwitterRobertaSentiment: wraps tokenizer+model, provides predict_proba/labels.

Inputs
------
- Python list[str]: the review texts to analyze.

Outputs
-------
- numpy.ndarray of shape (N, 3): per-class probabilities.
- Optionally class indices/labels via helper methods.

Typical use
-----------
from sentiment_infer import TwitterRobertaSentiment, InferenceConfig

cfg = InferenceConfig(max_length=256, chunk_long=True, agg="mean", batch_size=32, fp16=True)
clf = TwitterRobertaSentiment(cfg)
probs = clf.predict_proba(["Great camera!", "Meh.", "Terrible quality."])

Notes
-----
- Uses GPU automatically if available; falls back to CPU otherwise.
- fp16 gives speedups on modern NVIDIA GPUs; set fp16=False for CPU.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]

@dataclass
class InferenceConfig:
    model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 256
    batch_size: int = 32
    fp16: bool = True
    chunk_long: bool = True
    stride: int = 64
    agg: str = "mean"  # "mean" | "max"

class TwitterRobertaSentiment:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (self.device.type == "cuda" and self.cfg.fp16) else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id)
        self.model.to(self.device, dtype=dtype)
        self.model.eval()

    @torch.no_grad()
    def _forward_logits(self, encodings: Dict[str, torch.Tensor]) -> np.ndarray:
        enc = {k: v.to(self.device) for k, v in encodings.items()}
        logits = self.model(**enc).logits
        probs = softmax(logits, dim=-1).float().cpu().numpy()
        return probs

    def _batchify(self, items: List[Dict[str, List[int]]], batch_size: int):
        """
        items: list of dicts like {"input_ids": [..], "attention_mask": [..]} (chunked path)
        """
        for i in range(0, len(items), batch_size):
            batch = {k: [dic[k] for dic in items[i:i+batch_size]] for k in items[0].keys()}
            batch_padded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
            yield batch_padded

    def _predict_simple(self, texts: List[str]) -> np.ndarray:
        """
        Correct batching for the simple path:
        - Tokenize each mini-batch with padding=True and return_tensors="pt"
        - Feed tensors directly to the model
        """
        probs_all = []
        for i in range(0, len(texts), self.cfg.batch_size):
            chunk = texts[i:i+self.cfg.batch_size]
            batch = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            probs = self._forward_logits(batch)
            probs_all.append(probs)
        return np.vstack(probs_all) if probs_all else np.zeros((0, 3), dtype=np.float32)

    def _predict_chunked(self, texts: List[str]) -> np.ndarray:
        # Tokenize with overflowing chunks + stride
        tok = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.cfg.max_length,
            stride=self.cfg.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
        )
        overflow_to_sample = tok.pop("overflow_to_sample_mapping")
        # group encodings per original sample
        per_sample_ids = [[] for _ in texts]
        for i, sidx in enumerate(overflow_to_sample):
            per_sample_ids[sidx].append(
                {k: tok[k][i] for k in ["input_ids", "attention_mask"] if k in tok}
            )

        final_probs = []
        for group in per_sample_ids:
            probs_chunks = []
            if not group:
                final_probs.append(np.array([1/3, 1/3, 1/3], dtype=np.float32))
                continue
            for batch in self._batchify(group, self.cfg.batch_size):
                probs_chunks.append(self._forward_logits(batch))
            probs_chunks = np.vstack(probs_chunks)
            if self.cfg.agg == "max":
                agg_probs = probs_chunks.max(axis=0)
            else:
                agg_probs = probs_chunks.mean(axis=0)
            agg_probs = agg_probs / agg_probs.sum()
            final_probs.append(agg_probs)
        return np.vstack(final_probs)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.cfg.chunk_long:
            return self._predict_chunked(texts)
        return self._predict_simple(texts)

    def predict_label_ids(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        return probs.argmax(axis=1)

    def predict_labels(self, texts: List[str]) -> List[str]:
        idx = self.predict_label_ids(texts)
        return [LABELS[i] for i in idx]

