import os
import json

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

IMG_SIZE = 128

CHANNELS = 3

BATCH_SIZE = 2

RANDOM_STATE = 42


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REAL_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "processed_sequences",
    "real"
)

FAKE_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "processed_sequences",
    "fake"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "deepfake_model.h5"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "metrics.json"
)


# ============================================================
# LOAD DATASET PATHS
# ============================================================

def get_dataset():

    paths = []
    labels = []

    if os.path.exists(
        REAL_DIR
    ):

        for filename in os.listdir(
            REAL_DIR
        ):

            if filename.endswith(
                ".npy"
            ):

                paths.append(
                    os.path.join(
                        REAL_DIR,
                        filename
                    )
                )

                labels.append(0)

    if os.path.exists(
        FAKE_DIR
    ):

        for filename in os.listdir(
            FAKE_DIR
        ):

            if filename.endswith(
                ".npy"
            ):

                paths.append(
                    os.path.join(
                        FAKE_DIR,
                        filename
                    )
                )

                labels.append(1)

    return (
        np.asarray(paths),
        np.asarray(labels)
    )


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences(paths):

    sequences = []

    valid_paths = []

    for path in paths:

        try:

            sequence = np.load(
                path
            ).astype(
                np.float32
            )

            if sequence.shape != (
                SEQUENCE_LENGTH,
                IMG_SIZE,
                IMG_SIZE,
                CHANNELS
            ):

                continue

            sequence /= 255.0

            sequences.append(
                sequence
            )

            valid_paths.append(
                path
            )

        except Exception as e:

            print(
                f"Skipping {path}: {e}"
            )

    return np.asarray(
        sequences,
        dtype=np.float32
    ), valid_paths


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEEPFAKE MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------
    # Dataset
    # --------------------------------------------

    paths, labels = get_dataset()

    print(
        f"Total samples: {len(paths)}"
    )

    # --------------------------------------------
    # Same test split as training
    # --------------------------------------------

    (
        _,
        test_paths,
        _,
        test_labels
    ) = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(
        f"Test samples: {len(test_paths)}"
    )

    # --------------------------------------------
    # Load
    # --------------------------------------------

    X_test, valid_paths = load_sequences(
        test_paths
    )

    # Match labels to successfully loaded files
    valid_set = set(valid_paths)

    y_test = []

    for path in valid_paths:

        if path in test_paths:

            index = list(
                test_paths
            ).index(path)

            y_test.append(
                test_labels[index]
            )

    y_test = np.asarray(
        y_test
    )

    print(
        f"Valid test samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------
    # Model
    # --------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------
    # Predict
    # --------------------------------------------

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # --------------------------------------------
    # Save metrics
    # --------------------------------------------

    metrics = {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),

        "test_samples": int(
            len(y_test)
        )
    }

    os.makedirs(
        os.path.dirname(
            METRICS_PATH
        ),
        exist_ok=True
    )

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "\nMetrics saved to:"
    )

    print(
        METRICS_PATH
    )