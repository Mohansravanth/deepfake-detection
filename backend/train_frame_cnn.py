"""Train a frame-level CNN on face crops; aggregate per video for evaluation.

Uses the grouped v2 split: train on train-split videos only, select nothing on
test. Frame predictions are averaged per video (stride-2 frames, 10 per video)
to produce video-level scores comparable to the rest of the ensemble.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_PATH = BASE_DIR / "backend" / "experimental_sequences" / "v2_split.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "frame_cnn.keras"
METRICS_PATH = MODEL_DIR / "metrics_framecnn.json"
FRAME_STRIDE = 2
SEED = 42


def load_split_frames(split_name):
    rows = [r for r in csv.DictReader(open(SPLIT_PATH, encoding="utf-8", newline="")) if r["split"] == split_name]
    xs, ys, owners = [], [], []
    for i, row in enumerate(rows):
        seq = np.load(BASE_DIR / row["sequence_path"].replace("\\", "/"), allow_pickle=False)
        seq = seq[::FRAME_STRIDE]
        seq = seq.astype(np.float32)
        if seq.max() <= 1.5:
            seq = seq * 255.0
        xs.append(np.clip(seq, 0, 255).astype(np.uint8))
        label = int(row["label"])
        ys.extend([label] * len(seq))
        owners.extend([i] * len(seq))
        if (i + 1) % 100 == 0:
            print(f"{split_name}: loaded {i + 1}/{len(rows)}", flush=True)
    return np.concatenate(xs), np.array(ys, dtype=np.int32), np.array(owners), rows


def main():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train_f, _, _ = load_split_frames("train")
    X_val, y_val_f, val_owners, val_rows = load_split_frames("validation")
    X_test, y_test_f, test_owners, test_rows = load_split_frames("test")
    print("train frames:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape, flush=True)

    n_val_vids = len(val_rows)
    n_test_vids = len(test_rows)

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(128, 128, 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 1)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="frame_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # video-level early-stopping metric via callback on val frame loss is weak;
    # use val frame loss patience and evaluate video-level afterwards.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train_f,
        validation_data=(X_val, y_val_f),
        epochs=25,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_PATH)

    def video_probs(X, owners, n_videos):
        frame_probs = model.predict(X, verbose=0).reshape(-1)
        out = np.zeros(n_videos, dtype=np.float32)
        for v in range(n_videos):
            mask = owners == v
            if mask.any():
                out[v] = frame_probs[mask].mean()
        return out

    val_prob = video_probs(X_val, val_owners, n_val_vids)
    test_prob = video_probs(X_test, test_owners, n_test_vids)
    y_val = np.array([int(r["label"]) for r in val_rows])
    y_test = np.array([int(r["label"]) for r in test_rows])

    print("\nval video AUC:", roc_auc_score(y_val, val_prob))
    print("test video AUC:", roc_auc_score(y_test, test_prob))

    # threshold on val maximizing balanced recall + f1 (same rule as evaluate_ensemble)
    best = (0.0, 0.5)
    for th in np.linspace(0.05, 0.95, 181):
        preds = (val_prob >= th).astype(int)
        f1 = float(f1_score(y_val, preds, zero_division=0))
        rec_f = float(recall_score(y_val, preds, zero_division=0))
        rec_r = float(recall_score(1 - y_val, 1 - preds, zero_division=0))
        score = min(rec_f, rec_r) + 0.2 * f1
        if score > best[0]:
            best = (score, float(th))
    threshold = best[1]

    preds = (test_prob >= threshold).astype(int)
    m = confusion_matrix(y_test, preds, labels=[0, 1])
    metrics = {
        "model": "frame_cnn_v1",
        "frame_stride": FRAME_STRIDE,
        "threshold": threshold,
        "val_video_auc": float(roc_auc_score(y_val, val_prob)),
        "test_video_auc": float(roc_auc_score(y_test, test_prob)),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds, zero_division=0)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "confusion_matrix": m.tolist(),
        "test_samples": int(len(y_test)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    # save per-video probs for ensemble evaluation
    probs_path = BASE_DIR / "backend" / "v2_feature_cache" / "framecnn_video_probs.npz"
    np.savez(
        probs_path,
        val_prob=val_prob, test_prob=test_prob,
        val_labels=y_val, test_labels=y_test,
    )
    print("saved", MODEL_PATH, "and", probs_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
