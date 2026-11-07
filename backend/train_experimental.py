
import os
import csv
import json
import random
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"
SPLIT_PATH = DATA_DIR / "experimental_split.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "deepfake_model_experimental.keras"
METRICS_PATH = MODEL_DIR / "metrics_experimental.json"
HISTORY_PATH = MODEL_DIR / "training_history_experimental.json"

SEQUENCE_LENGTH = 20
IMG_SIZE = 128
CHANNELS = 3

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.0001

RANDOM_STATE = 42

EXPECTED_SHAPE = (
    SEQUENCE_LENGTH,
    IMG_SIZE,
    IMG_SIZE,
    CHANNELS
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("\n" + "=" * 70)
print("EXPERIMENTAL DEEPFAKE MODEL TRAINING")
print("=" * 70)

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

# ============================================================
# 3. LOAD TRAIN / VALIDATION / TEST MANIFEST
# ============================================================

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset split file not found: {SPLIT_PATH}"
    )

with open(SPLIT_PATH, "r", newline="", encoding="utf-8") as file:
    all_rows = list(csv.DictReader(file))

train_rows = [
    row for row in all_rows
    if row["split"] == "train"
]

validation_rows = [
    row for row in all_rows
    if row["split"] == "validation"
]

test_rows = [
    row for row in all_rows
    if row["split"] == "test"
]

if not train_rows or not validation_rows or not test_rows:
    raise ValueError("One or more dataset splits are empty.")

print("\nDATASET DISTRIBUTION")
print("-" * 70)

for name, rows in [
    ("TRAIN", train_rows),
    ("VALIDATION", validation_rows),
    ("TEST", test_rows)
]:

    counts = Counter(int(row["label"]) for row in rows)

    print(
        f"{name}: {len(rows)} samples | "
        f"Real: {counts[0]} | Fake: {counts[1]}"
    )

# ============================================================
# 4. VERIFY DATASET PATHS
# ============================================================

for row in all_rows:

    relative_path = row["sequence_path"].replace("\\", os.sep)
    sequence_path = BASE_DIR / relative_path

    if not sequence_path.is_file():
        raise FileNotFoundError(
            f"Missing sequence file: {sequence_path}"
        )

print("\nAll manifest sequence paths exist.")

# ============================================================
# 5. BATCH DATA GENERATOR
# ============================================================

class VideoSequenceGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        rows,
        batch_size=2,
        shuffle=False
    ):

        super().__init__()

        self.rows = list(rows)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.labels = np.array(
            [int(row["label"]) for row in self.rows],
            dtype=np.float32
        )

        self.indices = np.arange(len(self.rows))

        self.on_epoch_end()

    def __len__(self):

        return int(
            np.ceil(len(self.rows) / self.batch_size)
        )

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min(
            start + self.batch_size,
            len(self.rows)
        )

        batch_indices = self.indices[start:end]

        batch_x = []
        batch_y = []

        for sample_index in batch_indices:

            row = self.rows[sample_index]

            relative_path = row["sequence_path"].replace(
                "\\",
                os.sep
            )

            sequence_path = BASE_DIR / relative_path

            sequence = np.load(
                sequence_path,
                allow_pickle=False
            )

            if sequence.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Invalid sequence shape in {sequence_path}: "
                    f"{sequence.shape}"
                )

            sequence = sequence.astype(
                np.float32,
                copy=False
            )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"Non-finite values in {sequence_path}"
                )

            if sequence.min() < 0.0 or sequence.max() > 1.0:
                raise ValueError(
                    f"Invalid pixel range in {sequence_path}"
                )

            batch_x.append(sequence)
            batch_y.append(int(row["label"]))

        batch_x = np.stack(
            batch_x,
            axis=0
        ).astype(np.float32)

        batch_y = np.asarray(
            batch_y,
            dtype=np.float32
        )

        return batch_x, batch_y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# 6. CREATE DATA GENERATORS
# ============================================================

train_generator = VideoSequenceGenerator(
    train_rows,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = VideoSequenceGenerator(
    validation_rows,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = VideoSequenceGenerator(
    test_rows,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nBATCH GENERATOR INFORMATION")
print("-" * 70)

print("Training batches:", len(train_generator))
print("Validation batches:", len(validation_generator))
print("Testing batches:", len(test_generator))

# ============================================================
# 7. BUILD CNN-LSTM MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=EXPECTED_SHAPE,
        name="video_input"
    )

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )(inputs)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        )
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.MaxPooling2D(
            (2, 2)
        )
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.GlobalAveragePooling2D()
    )(x)

    x = tf.keras.layers.LSTM(128)(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fake_probability"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Experimental_Deepfake_CNN_LSTM"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    return model


model = build_model()

print("\nMODEL ARCHITECTURE")
print("-" * 70)

model.summary()

# ============================================================
# 8. TRAINING CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]

# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING EXPERIMENTAL TRAINING")
print("=" * 70)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w", encoding="utf-8") as file:

    json.dump(
        history_data,
        file,
        indent=4
    )

print("\nTraining history saved:", HISTORY_PATH)

# ============================================================
# 11. LOAD BEST VALIDATION MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Best experimental model was not saved."
    )

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nBest model loaded from:", MODEL_PATH)

# ============================================================
# 12. FINAL HELD-OUT TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

test_results = best_model.evaluate(
    test_generator,
    verbose=1,
    return_dict=True
)

predicted_probabilities = best_model.predict(
    test_generator,
    verbose=1
).reshape(-1)

actual_labels = test_generator.labels.astype(int)

predicted_labels = (
    predicted_probabilities >= 0.5
).astype(int)

# ============================================================
# 13. CALCULATE CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

roc_auc = roc_auc_score(
    actual_labels,
    predicted_probabilities
)

pr_auc = average_precision_score(
    actual_labels,
    predicted_probabilities
)

confusion = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=[0, 1]
)

report = classification_report(
    actual_labels,
    predicted_labels,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    output_dict=True,
    zero_division=0
)

# ============================================================
# 14. SAVE EXPERIMENTAL METRICS
# ============================================================

metrics = {

    "model": "Experimental CNN-LSTM",

    "test_samples": int(len(actual_labels)),

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "roc_auc": float(roc_auc),

    "pr_auc": float(pr_auc),

    "confusion_matrix": confusion.tolist(),

    "classification_report": report,

    "keras_test_metrics": {
        key: float(value)
        for key, value in test_results.items()
    },

    "prediction_threshold": 0.5,

    "class_mapping": {
        "0": "REAL",
        "1": "FAKE"
    },

    "dataset_split": {
        "training_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "testing_samples": len(test_rows)
    }
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

# ============================================================
# 15. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTAL MODEL RESULTS")
print("=" * 70)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print(f"PR AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix")
print("[[True Real, False Fake],")
print(" [False Real, True Fake]]")

print(confusion)

print("\nClassification Report")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0
    )
)

print("\nEXPERIMENTAL FILES")
print("-" * 70)

print("Model:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("History:", HISTORY_PATH)

print("\nEXPERIMENTAL TRAINING COMPLETED")
print("=" * 70)
