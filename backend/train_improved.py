import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
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
FEATURE_CACHE = BASE_DIR / "backend" / "v2_feature_cache" / "mobilenetv2_features.npy"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "improved_detector.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METRICS_COPY = BASE_DIR / "backend" / "data" / "metrics.json"
METRICS_COPY.parent.mkdir(parents=True, exist_ok=True)


def load_split():
    with open(SPLIT_PATH, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("v2_split.csv is empty.")
    return rows


def build_forensic_matrix(rows):
    features = []
    labels = []
    for index, row in enumerate(rows, start=1):
        relative = row["sequence_path"].replace("\\", "/")
        path = BASE_DIR / relative
        sequence = np.load(path, allow_pickle=False)
        features.append(extract_forensic_vector(sequence, face_coverage=0.55))
        labels.append(int(row["label"]))
        if index % 50 == 0:
            print(f"Forensic features {index}/{len(rows)}", flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def split_indices(rows):
    groups = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(rows):
        groups[row["split"]].append(index)
    return groups


def choose_threshold(y_true, probabilities):
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in np.linspace(0.15, 0.85, 71):
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        # Prefer balanced real/fake recall, then F1.
        real_recall = recall_score(1 - y_true, 1 - preds, zero_division=0)
        score = min(recall, real_recall) + 0.15 * f1
        if score > best[0]:
            best = (score, float(threshold), float(precision), float(recall))
    return best[1]


def evaluate(y_true, probabilities, threshold):
    preds = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "test_real": int((y_true == 0).sum()),
        "test_fake": int((y_true == 1).sum()),
        "test_samples": int(len(y_true)),
        "threshold": float(threshold),
    }


def main():
    print("Training improved real/fake detector...", flush=True)
    rows = load_split()
    forensic_x, y = build_forensic_matrix(rows)
    indices = split_indices(rows)

    embedding_x = None
    pca = None
    if FEATURE_CACHE.exists():
        embeddings = np.load(FEATURE_CACHE, mmap_mode="r")
        if embeddings.shape[0] == len(rows):
            pooled = np.concatenate(
                [
                    embeddings.mean(axis=1),
                    embeddings.std(axis=1),
                    np.abs(np.diff(embeddings, axis=1)).mean(axis=1),
                ],
                axis=1,
            )
            pca = PCA(n_components=48, random_state=42)
            train_idx = indices["train"]
            pca.fit(pooled[train_idx])
            embedding_x = pca.transform(pooled)
            print("Using MobileNet pooled embeddings.", flush=True)

    if embedding_x is not None:
        combined = np.concatenate([forensic_x, embedding_x.astype(np.float32)], axis=1)
    else:
        combined = forensic_x

    x_train, y_train = combined[indices["train"]], y[indices["train"]]
    x_val, y_val = combined[indices["validation"]], y[indices["validation"]]
    x_test, y_test = combined[indices["test"]], y[indices["test"]]

    forensic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=250,
                    min_samples_leaf=24,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    embedding_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.08,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    forensic_model.fit(forensic_x[indices["train"]], y_train)
    embedding_model.fit(x_train, y_train)

    val_forensic = forensic_model.predict_proba(forensic_x[indices["validation"]])[:, 1]
    val_embed = embedding_model.predict_proba(x_val)[:, 1]
    forensic_auc = roc_auc_score(y_val, val_forensic)
    embed_auc = roc_auc_score(y_val, val_embed)
    # Website inference uses forensic features only so it stays fast and stable.
    # Keep the combined model only when it clearly beats forensics on validation.
    if embed_auc > forensic_auc + 0.03:
        best_blend = 0.35
        val_prob = best_blend * val_forensic + (1.0 - best_blend) * val_embed
        best_auc = roc_auc_score(y_val, val_prob)
        test_prob = (
            best_blend * forensic_model.predict_proba(forensic_x[indices["test"]])[:, 1]
            + (1.0 - best_blend) * embedding_model.predict_proba(x_test)[:, 1]
        )
        uses_embeddings = pca is not None
    else:
        best_blend = 1.0
        val_prob = val_forensic
        best_auc = forensic_auc
        test_prob = forensic_model.predict_proba(forensic_x[indices["test"]])[:, 1]
        uses_embeddings = False
    threshold = choose_threshold(y_val, val_prob)
    metrics = evaluate(y_test, test_prob, threshold)
    metrics.update(
        {
            "model": "Forensic + MobileNet ensemble",
            "blend_forensic_weight": best_blend,
            "validation_auc": float(best_auc),
            "class_mapping": {"0": "REAL", "1": "FAKE"},
        }
    )

    payload = {
        "forensic_model": forensic_model,
        "embedding_model": embedding_model,
        "pca": pca,
        "blend_forensic_weight": best_blend,
        "threshold": threshold,
        "uses_embeddings": pca is not None,
        "feature_names": list(range(combined.shape[1])),
    }
    joblib.dump(payload, MODEL_PATH)

    for path in (METRICS_PATH, METRICS_COPY):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved {MODEL_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
