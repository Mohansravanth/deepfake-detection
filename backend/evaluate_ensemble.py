"""Offline ensemble evaluation on the grouped v2 split.

Signals evaluated per sequence:
  - forensic: HistGradientBoosting on forensic_features vector (improved_detector.joblib)
  - vit_mean / vit_head: pretrained ViT per-frame probs, aggregated directly
    or through a small temporal head trained on the train split only
  - tf_seq: sequence_classifier_v2 on cached MobileNetV2 embeddings

Blend weights + decision threshold are selected on the validation split and
reported on the untouched test split. Results are written to
models/ensemble_config.json for app.py to consume.
"""

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forensic_features import extract_forensic_vector

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_PATH = BASE_DIR / "backend" / "experimental_sequences" / "v2_split.csv"
CACHE_DIR = BASE_DIR / "backend" / "v2_feature_cache"
FORENSIC_CACHE = CACHE_DIR / "forensic_features.npy"
VIT_PROBS = CACHE_DIR / "vit_frame_probs.npy"
MOBILENET_FEATURES = CACHE_DIR / "mobilenetv2_features.npy"
SEQ_CLASSIFIER = BASE_DIR / "models" / "sequence_classifier_v2.keras"
IMPROVED_BUNDLE = BASE_DIR / "models" / "improved_detector.joblib"
CONFIG_PATH = BASE_DIR / "models" / "ensemble_config.json"
METRICS_OUT = BASE_DIR / "models" / "metrics_ensemble.json"


def load_rows():
    with open(SPLIT_PATH, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def split_indices(rows):
    groups = {"train": [], "validation": [], "test": []}
    for i, row in enumerate(rows):
        groups[row["split"]].append(i)
    return groups


def forensic_matrix(rows):
    if FORENSIC_CACHE.exists():
        cached = np.load(FORENSIC_CACHE)
        if cached.shape[0] == len(rows):
            print("Using cached forensic features.")
            return cached
    feats = np.zeros((len(rows), 24), dtype=np.float32)
    for i, row in enumerate(rows):
        seq = np.load(BASE_DIR / row["sequence_path"].replace("\\", "/"), allow_pickle=False)
        feats[i] = extract_forensic_vector(seq, face_coverage=0.55)
        if (i + 1) % 50 == 0:
            print(f"forensic {i + 1}/{len(rows)}", flush=True)
    np.save(FORENSIC_CACHE, feats)
    return feats


def vit_stats(probs):
    """(N,20) per-frame probs -> summary stats."""
    t = np.arange(probs.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    slope = (probs * t).sum(axis=1) / (t * t).sum()
    return np.stack(
        [
            probs.mean(axis=1),
            probs.std(axis=1),
            probs.min(axis=1),
            probs.max(axis=1),
            np.median(probs, axis=1),
            np.mean(probs > 0.5, axis=1),
            slope,
            probs[:, -1] - probs[:, 0],
        ],
        axis=1,
    ).astype(np.float32)


def choose_threshold(y_true, probs):
    """Vectorized: pick threshold maximizing min(recall_fake, recall_real) + 0.2*f1."""
    ths = np.linspace(0.05, 0.95, 181)
    preds = probs[None, :] >= ths[:, None]  # (T, N)
    y = y_true.astype(bool)[None, :]
    tp = (preds & y).sum(axis=1)
    fp = (preds & ~y).sum(axis=1)
    fn = (~preds & y).sum(axis=1)
    tn = (~preds & ~y).sum(axis=1)
    rec_fake = tp / np.maximum(tp + fn, 1)
    rec_real = tn / np.maximum(tn + fp, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    score = np.minimum(rec_fake, rec_real) + 0.2 * f1
    return float(ths[int(np.argmax(score))])


def evaluate(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    m = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, probs)),
        "confusion_matrix": m.tolist(),
    }


def main():
    rows = load_rows()
    idx = split_indices(rows)
    y = np.array([r["label"] for r in rows], dtype=np.int32)
    y_val, y_test = y[idx["validation"]], y[idx["test"]]

    signals = {}

    # --- forensic ---
    if not IMPROVED_BUNDLE.exists():
        print("improved_detector.joblib missing - run train_improved.py first")
        return 1
    bundle = joblib.load(IMPROVED_BUNDLE)
    fx = forensic_matrix(rows)
    forensic_prob = bundle["forensic_model"].predict_proba(fx)[:, 1]
    signals["forensic"] = forensic_prob
    print("forensic val AUC:", round(roc_auc_score(y_val, forensic_prob[idx["validation"]]), 4))

    # --- frame cnn (video-level probs saved by train_frame_cnn.py) ---
    framecnn_path = CACHE_DIR / "framecnn_video_probs.npz"
    if framecnn_path.exists():
        d = np.load(framecnn_path)
        fc = np.full(len(rows), np.nan, dtype=np.float32)
        vi_rows = [i for i, r in enumerate(rows) if r["split"] == "validation"]
        ti_rows = [i for i, r in enumerate(rows) if r["split"] == "test"]
        fc[vi_rows] = d["val_prob"]
        fc[ti_rows] = d["test_prob"]
        signals["framecnn"] = fc

    # --- vit frame probs (optional; skipped if cache missing) ---
    if VIT_PROBS.exists():
        vit_probs = np.load(VIT_PROBS)
        signals["vit_mean"] = vit_probs.mean(axis=1)

        stats = vit_stats(vit_probs)
        head = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
            ]
        )
        head.fit(stats[idx["train"]], y[idx["train"]])
        signals["vit_head"] = head.predict_proba(stats)[:, 1]
        joblib.dump(head, CACHE_DIR / "vit_temporal_head.joblib")

        from sklearn.ensemble import HistGradientBoostingClassifier

        gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)
        gbm.fit(vit_probs[idx["train"]], y[idx["train"]])
        signals["vit_gbm"] = gbm.predict_proba(vit_probs)[:, 1]
        joblib.dump(gbm, CACHE_DIR / "vit_frame_gbm.joblib")
    else:
        print("vit_frame_probs.npy missing - skipping vit signals")

    # --- tf sequence classifier ---
    if SEQ_CLASSIFIER.exists() and MOBILENET_FEATURES.exists():
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        feats = np.load(MOBILENET_FEATURES, mmap_mode="r")
        clf = tf.keras.models.load_model(SEQ_CLASSIFIER, compile=False)
        tf_prob = np.zeros(len(rows), dtype=np.float32)
        for name, indices in idx.items():
            preds = clf.predict(np.asarray(feats[indices], dtype=np.float32), verbose=0)
            tf_prob[indices] = np.asarray(preds).reshape(-1)
        signals["tf_seq"] = tf_prob
        del clf

    # --- report individual signal quality on validation ---
    print("\n=== Validation AUC per signal ===")
    vi = idx["validation"]
    for name, prob in signals.items():
        try:
            auc = roc_auc_score(y_val, prob[vi])
        except Exception:
            auc = float("nan")
        print(f"{name:10} AUC={auc:.4f} mean_prob_real={prob[vi][y_val==0].mean():.3f} fake={prob[vi][y_val==1].mean():.3f}")

    # --- grid-search blend on validation ---
    names = list(signals.keys())
    best = None
    step = 0.05
    grid = np.arange(0, 1 + step, step)

    def compositions(k):
        if k == 1:
            yield (1.0,)
            return
        for first in grid:
            for rest in compositions(k - 1):
                s = sum(rest)
                if s <= 1.0 + 1e-9:
                    yield (first,) + tuple(r * (1 - first) / s if s > 0 else r for r in rest)

    for k in (1, 2, 3):
        if k > len(names):
            continue
        for combo_idx in __import__("itertools").combinations(range(len(names)), k):
            combo_names = [names[i] for i in combo_idx]
            for weights in compositions(k):
                val_prob = sum(w * signals[n][vi] for w, n in zip(weights, combo_names))
                th = choose_threshold(y_val, val_prob)
                m = evaluate(y_val, val_prob, th)
                score = min(
                    m["confusion_matrix"][0][0] / max(m["confusion_matrix"][0][0] + m["confusion_matrix"][0][1], 1),
                    m["confusion_matrix"][1][1] / max(m["confusion_matrix"][1][0] + m["confusion_matrix"][1][1], 1),
                ) + 0.2 * m["f1_score"]
                if best is None or score > best[0]:
                    best = (score, dict(zip(combo_names, weights)), th, m)

    score, weights, threshold, val_metrics = best
    print("\n=== Best validation blend ===")
    print("weights:", {k: round(v, 3) for k, v in weights.items() if v > 1e-6})
    print("threshold:", round(threshold, 4))
    print(json.dumps(val_metrics, indent=2))

    # --- test evaluation with chosen blend ---
    ti = idx["test"]
    test_prob = sum(w * signals[n][ti] for n, w in weights.items())
    test_metrics = evaluate(y_test, test_prob, threshold)
    print("\n=== TEST ===")
    print(json.dumps(test_metrics, indent=2))

    config = {
        "signals": {k: round(float(v), 4) for k, v in weights.items() if v > 1e-6},
        "threshold": float(threshold),
        "validation": val_metrics,
        "test": test_metrics,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    METRICS_OUT.write_text(json.dumps({"ensemble": config}, indent=2), encoding="utf-8")
    print("\nSaved", CONFIG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
