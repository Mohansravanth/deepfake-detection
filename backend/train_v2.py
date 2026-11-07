
import csv
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from collections import Counter

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Rescaling
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = BASE_DIR / "backend" / "experimental_sequences"

SPLIT_PATH = SEQUENCE_DIR / "v2_split.csv"

FEATURE_DIR = BASE_DIR / "backend" / "v2_feature_cache"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CACHE = FEATURE_DIR / "mobilenetv2_features.npy"
FEATURE_METADATA = FEATURE_DIR / "feature_metadata.json"

CLASSIFIER_PATH = MODEL_DIR / "sequence_classifier_v2.keras"
FINAL_MODEL_PATH = MODEL_DIR / "deepfake_model_v2.keras"
METRICS_PATH = MODEL_DIR / "metrics_v2.json"
HISTORY_PATH = MODEL_DIR / "training_history_v2.json"

SEQUENCE_LENGTH = 20
IMAGE_SIZE = 128
FEATURE_SIZE = 1280

FEATURE_EXTRACTION_BATCH_VIDEOS = 2
FEATURE_EXTRACTION_FRAME_BATCH = 8

EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 0.001

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================================================
# GPU INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("DEEPFAKE DETECTOR V2")
print("MobileNetV2 + Bidirectional LSTM")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# =========================================================
# LOAD DATASET SPLIT
# =========================================================

with open(SPLIT_PATH, "r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("V2 split file is empty.")

for row in rows:
    row["label"] = int(row["label"])

print("\nDATASET SUMMARY")
print("-" * 70)

for split_name in ["train", "validation", "test"]:

    selected = [
        row for row in rows
        if row["split"] == split_name
    ]

    counts = Counter(row["label"] for row in selected)

    print(
        f"{split_name.upper():12} "
        f"Total={len(selected):4} "
        f"REAL={counts[0]:4} "
        f"FAKE={counts[1]:4}"
    )

# =========================================================
# LOAD PRETRAINED MOBILENETV2
# =========================================================

print("\nLoading pretrained MobileNetV2...")

backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

backbone.trainable = False

print("MobileNetV2 loaded.")
print("Feature output:", backbone.output_shape)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features():

    print("\nExtracting video frame features...")

    total = len(rows)

    features = np.lib.format.open_memmap(
        FEATURE_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(total, SEQUENCE_LENGTH, FEATURE_SIZE)
    )

    processed = 0

    for start in range(
        0,
        total,
        FEATURE_EXTRACTION_BATCH_VIDEOS
    ):

        batch_rows = rows[
            start:
            start + FEATURE_EXTRACTION_BATCH_VIDEOS
        ]

        sequences = []

        for row in batch_rows:

            relative_path = row["sequence_path"].replace("\\", "/")

            sequence_path = BASE_DIR / relative_path

            if not sequence_path.exists():
                raise FileNotFoundError(sequence_path)

            sequence = np.load(
                sequence_path,
                allow_pickle=False
            )

            if sequence.shape != (
                SEQUENCE_LENGTH,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            ):
                raise ValueError(
                    f"Invalid sequence shape: "
                    f"{sequence_path} {sequence.shape}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"Non-finite values found: {sequence_path}"
                )

            if sequence.min() < -0.001 or sequence.max() > 1.001:
                raise ValueError(
                    f"Invalid pixel range: {sequence_path}"
                )

            sequences.append(sequence)

        batch = np.stack(sequences).astype(np.float32)

        video_count = len(batch)

        # [0, 1] -> [-1, 1]
        batch = batch * 2.0 - 1.0

        # Process frames without loading the entire dataset into RAM.
        flattened = batch.reshape(
            video_count * SEQUENCE_LENGTH,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )

        extracted = backbone.predict(
            flattened,
            batch_size=FEATURE_EXTRACTION_FRAME_BATCH,
            verbose=0
        )

        extracted = extracted.reshape(
            video_count,
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        )

        features[
            start:start + video_count
        ] = extracted

        processed += video_count

        print(
            f"Processed videos: {processed}/{total}"
        )

        del batch
        del flattened
        del extracted
        del sequences

    features.flush()

    metadata = {
        "model": "MobileNetV2",
        "weights": "imagenet",
        "sequence_length": SEQUENCE_LENGTH,
        "image_size": IMAGE_SIZE,
        "feature_size": FEATURE_SIZE,
        "total_sequences": total,
        "input_normalization": "0_to_1_then_minus1_to_1",
        "split_file": str(SPLIT_PATH)
    }

    with open(
        FEATURE_METADATA,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nFeature extraction completed.")
    print("Feature cache:", FEATURE_CACHE)

    return features


# =========================================================
# LOAD OR CREATE FEATURE CACHE
# =========================================================

expected_shape = (
    len(rows),
    SEQUENCE_LENGTH,
    FEATURE_SIZE
)

cache_valid = False

if FEATURE_CACHE.exists() and FEATURE_METADATA.exists():

    try:

        with open(
            FEATURE_METADATA,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        cached = np.load(
            FEATURE_CACHE,
            mmap_mode="r"
        )

        cache_valid = (
            cached.shape == expected_shape
            and metadata.get("total_sequences") == len(rows)
            and metadata.get("sequence_length") == SEQUENCE_LENGTH
            and metadata.get("image_size") == IMAGE_SIZE
            and metadata.get("feature_size") == FEATURE_SIZE
        )

        if cache_valid:
            print("\nExisting feature cache found.")
            print("Using cached features.")

    except Exception:
        cache_valid = False

if cache_valid:

    features = np.load(
        FEATURE_CACHE,
        mmap_mode="r"
    )

else:

    features = extract_features()

print("Feature array shape:", features.shape)

# =========================================================
# PREPARE TRAIN / VALIDATION / TEST ARRAYS
# =========================================================

train_indices = [
    index for index, row in enumerate(rows)
    if row["split"] == "train"
]

validation_indices = [
    index for index, row in enumerate(rows)
    if row["split"] == "validation"
]

test_indices = [
    index for index, row in enumerate(rows)
    if row["split"] == "test"
]

def prepare_data(indices):

    X = np.asarray(
        features[indices],
        dtype=np.float32
    )

    y = np.asarray(
        [rows[index]["label"] for index in indices],
        dtype=np.float32
    )

    return X, y


X_train, y_train = prepare_data(train_indices)
X_val, y_val = prepare_data(validation_indices)
X_test, y_test = prepare_data(test_indices)

print("\nFEATURE DATA SHAPES")
print("-" * 70)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)

class_weights = {
    0: float(weights[0]),
    1: float(weights[1])
}

print("\nCLASS WEIGHTS:", class_weights)

# =========================================================
# BUILD TEMPORAL CLASSIFIER
# =========================================================

print("\nBuilding Bidirectional LSTM classifier...")

sequence_input = Input(
    shape=(SEQUENCE_LENGTH, FEATURE_SIZE),
    name="sequence_features"
)

x = Bidirectional(
    LSTM(
        128,
        return_sequences=False,
        dropout=0.25,
        recurrent_dropout=0.0
    ),
    name="bidirectional_lstm"
)(sequence_input)

x = Dropout(0.4)(x)

x = Dense(
    64,
    activation="relu",
    name="dense_features"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    1,
    activation="sigmoid",
    name="fake_probability"
)(x)

classifier = Model(
    sequence_input,
    output,
    name="deepfake_temporal_classifier_v2"
)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)

classifier.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    ModelCheckpoint(
        filepath=str(CLASSIFIER_PATH),
        monitor="val_roc_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_roc_auc",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# =========================================================
# TRAIN
# =========================================================

print("\nStarting V2 training...")

history = classifier.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        file,
        indent=4
    )

# =========================================================
# LOAD BEST CLASSIFIER
# =========================================================

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH
)

# =========================================================
# SELECT THRESHOLD USING VALIDATION DATA ONLY
# =========================================================

val_probabilities = classifier.predict(
    X_val,
    verbose=0
).ravel()

threshold_results = []

for threshold in np.arange(0.20, 0.81, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    threshold_results.append(
        (float(f1), float(threshold), float(precision), float(recall))
    )

best_result = max(
    threshold_results,
    key=lambda item: (item[0], item[2])
)

best_threshold = best_result[1]

print("\nVALIDATION THRESHOLD SELECTION")
print("-" * 70)

print("Selected threshold:", round(best_threshold, 4))
print("Validation F1:", round(best_result[0], 4))
print("Validation precision:", round(best_result[2], 4))
print("Validation recall:", round(best_result[3], 4))

# =========================================================
# BUILD SINGLE VIDEO-INPUT MODEL
# =========================================================

print("\nBuilding combined video-input model...")

raw_video_input = Input(
    shape=(
        SEQUENCE_LENGTH,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),
    name="video_sequence"
)

# Match the preprocessing used during feature extraction.
normalized_input = Rescaling(
    scale=2.0,
    offset=-1.0,
    name="mobilenet_normalization"
)(raw_video_input)

temporal_features = tf.keras.layers.TimeDistributed(
    backbone,
    name="frame_feature_extractor"
)(normalized_input)

final_output = classifier(
    temporal_features
)

final_model = Model(
    raw_video_input,
    final_output,
    name="deepfake_detector_v2"
)

final_model.save(FINAL_MODEL_PATH)

print("Combined model saved:", FINAL_MODEL_PATH)

# =========================================================
# TEST SET EVALUATION
# =========================================================

print("\nEvaluating untouched test set...")

test_probabilities = classifier.predict(
    X_test,
    verbose=0
).ravel()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

specificity = (
    float(tn / (tn + fp))
    if (tn + fp) > 0
    else 0.0
)

# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "model": "MobileNetV2_Bidirectional_LSTM_V2",

    "test_samples": int(len(y_test)),

    "threshold": float(best_threshold),

    "accuracy": float(test_accuracy),

    "precision": float(test_precision),

    "recall": float(test_recall),

    "f1_score": float(test_f1),

    "specificity": float(specificity),

    "roc_auc": float(test_auc),

    "pr_auc": float(test_pr_auc),

    "confusion_matrix": matrix.tolist(),

    "true_negative": int(tn),

    "false_positive": int(fp),

    "false_negative": int(fn),

    "true_positive": int(tp),

    "test_real": int(np.sum(y_test == 0)),

    "test_fake": int(np.sum(y_test == 1)),

    "validation_threshold_selection": "Maximum validation F1; ties prefer precision",

    "pretrained_backbone": "MobileNetV2 ImageNet",

    "temporal_model": "Bidirectional LSTM"
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 70)
print("DEEPFAKE DETECTOR V2 - FINAL TEST RESULTS")
print("=" * 70)

print(f"Threshold   : {best_threshold:.4f}")
print(f"Accuracy    : {test_accuracy:.4f}")
print(f"Precision   : {test_precision:.4f}")
print(f"Recall      : {test_recall:.4f}")
print(f"F1 Score    : {test_f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"ROC AUC     : {test_auc:.4f}")
print(f"PR AUC      : {test_pr_auc:.4f}")

print("\nConfusion Matrix:")
print(matrix)

print("\nREAL videos correctly identified:", tn)
print("REAL videos incorrectly identified as FAKE:", fp)
print("FAKE videos incorrectly identified as REAL:", fn)
print("FAKE videos correctly identified:", tp)

print("\nSaved model:", FINAL_MODEL_PATH)
print("Saved metrics:", METRICS_PATH)
print("Saved history:", HISTORY_PATH)

print("\nV2 TRAINING AND EVALUATION COMPLETED")
