
import csv
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"

SPLIT_PATH = DATA_DIR / "experimental_split.csv"

MODEL_PATH = (
    BASE_DIR / "models" /
    "deepfake_model_experimental.keras"
)

HISTORY_PATH = (
    BASE_DIR / "models" /
    "training_history_experimental.json"
)

BATCH_SIZE = 2

print("\n" + "=" * 70)
print("EXPERIMENTAL MODEL DIAGNOSTICS")
print("=" * 70)

# --------------------------------------------------
# 1. Load training history
# --------------------------------------------------

with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)

print("\nTRAINING HISTORY")
print("-" * 70)

for key, values in history.items():

    print(f"\n{key}:")

    for epoch, value in enumerate(values, start=1):
        print(f"Epoch {epoch}: {value:.6f}")

# --------------------------------------------------
# 2. Load model and split manifest
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(SPLIT_PATH, "r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("\nMODEL INFORMATION")
print("-" * 70)

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters:", model.count_params())

# --------------------------------------------------
# 3. Calculate predictions in batches
# --------------------------------------------------

def inspect_split(split_name):

    selected = [
        row for row in rows
        if row["split"] == split_name
    ]

    labels = []
    probabilities = []

    for start in range(0, len(selected), BATCH_SIZE):

        batch_rows = selected[start:start + BATCH_SIZE]

        batch_sequences = []

        for row in batch_rows:

            relative_path = row["sequence_path"].replace("\\", "/")

            path = BASE_DIR / relative_path

            sequence = np.load(
                path,
                allow_pickle=False
            )

            batch_sequences.append(
                sequence.astype(np.float32, copy=False)
            )

            labels.append(int(row["label"]))

        batch_x = np.stack(batch_sequences).astype(np.float32)

        batch_predictions = model.predict(
            batch_x,
            verbose=0
        ).reshape(-1)

        probabilities.extend(batch_predictions.tolist())

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    print(f"\n{split_name.upper()} PREDICTIONS")
    print("-" * 70)

    for label, class_name in [
        (0, "REAL"),
        (1, "FAKE")
    ]:

        scores = probabilities[labels == label]

        print(f"\n{class_name}")
        print("Samples:", len(scores))
        print("Mean probability:", round(float(scores.mean()), 6))
        print("Std deviation:", round(float(scores.std()), 6))
        print("Minimum:", round(float(scores.min()), 6))
        print("Maximum:", round(float(scores.max()), 6))
        print("25th percentile:", round(float(np.percentile(scores, 25)), 6))
        print("50th percentile:", round(float(np.percentile(scores, 50)), 6))
        print("75th percentile:", round(float(np.percentile(scores, 75)), 6))

        print(
            "Predicted FAKE at 0.5:",
            int(np.sum(scores >= 0.5)),
            "/",
            len(scores)
        )

    auc = roc_auc_score(labels, probabilities)

    print("\nROC AUC:", round(float(auc), 6))

    print(
        "Overall probability mean:",
        round(float(probabilities.mean()), 6)
    )

# --------------------------------------------------
# 4. Inspect training and validation
# --------------------------------------------------

inspect_split("train")
inspect_split("validation")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETED")
print("=" * 70)
