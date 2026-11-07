
import csv
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

SPLIT_PATH = (
    BASE_DIR
    / "backend"
    / "experimental_sequences"
    / "experimental_split.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "deepfake_model_experimental.keras"
)

EXPECTED_SHAPE = (20, 128, 128, 3)
BATCH_SIZE = 2

print("\n" + "=" * 70)
print("SOURCE GROUP PREDICTION ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# Load manifest
# --------------------------------------------------

with open(SPLIT_PATH, "r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

rows = [
    row for row in rows
    if row["status"] == "processed"
]

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")
print("Input shape:", model.input_shape)

# --------------------------------------------------
# Analyze each split
# --------------------------------------------------

for split_name in ["train", "validation", "test"]:

    split_rows = [
        row for row in rows
        if row["split"] == split_name
    ]

    groups = defaultdict(list)

    for row in split_rows:
        groups[row["source_group"]].append(row)

    print("\n" + "=" * 70)
    print(split_name.upper(), "SOURCE GROUP RESULTS")
    print("=" * 70)

    group_results = []

    for group_name, group_rows in sorted(groups.items()):

        predictions = []
        actual_labels = []

        for start in range(0, len(group_rows), BATCH_SIZE):

            batch_rows = group_rows[
                start:start + BATCH_SIZE
            ]

            batch_sequences = []

            for row in batch_rows:

                relative_path = row["sequence_path"].replace(
                    "\\", "/"
                )

                sequence_path = BASE_DIR / relative_path

                sequence = np.load(
                    sequence_path,
                    allow_pickle=False
                )

                if sequence.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"Invalid shape: {sequence_path}"
                    )

                batch_sequences.append(sequence)

                actual_labels.append(int(row["label"]))

            batch_x = np.stack(
                batch_sequences
            ).astype(np.float32)

            batch_predictions = model.predict(
                batch_x,
                verbose=0
            ).reshape(-1)

            predictions.extend(
                batch_predictions.tolist()
            )

        predictions = np.asarray(predictions)
        actual_labels = np.asarray(actual_labels)

        mean_probability = float(np.mean(predictions))
        std_probability = float(np.std(predictions))

        predicted_fake = int(
            np.sum(predictions >= 0.5)
        )

        group_results.append(
            (
                group_name,
                int(actual_labels[0]),
                len(actual_labels),
                mean_probability,
                std_probability,
                predicted_fake
            )
        )

    print(
        f"\n{'GROUP':<13}"
        f"{'LABEL':<8}"
        f"{'COUNT':<8}"
        f"{'MEAN PROB':<13}"
        f"{'STD':<12}"
        f"{'FAKE PRED':<12}"
    )

    for result in group_results:

        name, label, count, mean, std, fake_count = result

        class_name = "REAL" if label == 0 else "FAKE"

        print(
            f"{name:<13}"
            f"{class_name:<8}"
            f"{count:<8}"
            f"{mean:<13.6f}"
            f"{std:<12.6f}"
            f"{fake_count}/{count:<12}"
        )

    # --------------------------------------------------
    # Class-level comparison
    # --------------------------------------------------

    real_groups = [
        result for result in group_results
        if result[1] == 0
    ]

    fake_groups = [
        result for result in group_results
        if result[1] == 1
    ]

    print("\nGROUP-LEVEL SUMMARY")

    if real_groups:
        print(
            "Real group average probability:",
            round(
                np.mean([result[3] for result in real_groups]),
                6
            )
        )

    if fake_groups:
        print(
            "Fake group average probability:",
            round(
                np.mean([result[3] for result in fake_groups]),
                6
            )
        )

print("\n" + "=" * 70)
print("GROUP ANALYSIS COMPLETED")
print("=" * 70)
