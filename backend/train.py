import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

IMG_SIZE = 128

CHANNELS = 3

BATCH_SIZE = 2

EPOCHS = 10

LEARNING_RATE = 0.0001

RANDOM_STATE = 42


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.join(BASE_DIR, "backend", "processed_sequences")
)

REAL_DIR = os.path.join(
    DATASET_DIR,
    "real"
)

FAKE_DIR = os.path.join(
    DATASET_DIR,
    "fake"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_model.h5"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    paths = []
    labels = []

    # REAL = 0
    if os.path.exists(REAL_DIR):

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

    # FAKE = 1
    if os.path.exists(FAKE_DIR):

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

    print(
        f"\nTotal samples: {len(paths)}"
    )

    print(
        f"Real samples: "
        f"{labels.count(0)}"
    )

    print(
        f"Fake samples: "
        f"{labels.count(1)}"
    )

    return paths, np.asarray(
        labels,
        dtype=np.float32
    )


# ============================================================
# LOAD NUMPY FILES
# ============================================================

def load_sequences(paths):

    X = []

    for index, path in enumerate(
        paths
    ):

        try:

            sequence = np.load(
                path
            )

            if sequence.shape != (
                SEQUENCE_LENGTH,
                IMG_SIZE,
                IMG_SIZE,
                CHANNELS
            ):

                print(
                    f"Skipping {path}"
                )

                print(
                    f"Shape: "
                    f"{sequence.shape}"
                )

                continue

            sequence = sequence.astype(
                np.float32
            )

            sequence /= 255.0

            X.append(
                sequence
            )

        except Exception as e:

            print(
                f"Error loading {path}: {e}"
            )

        if (
            index + 1
        ) % 50 == 0:

            print(
                f"Loaded "
                f"{index + 1}/"
                f"{len(paths)}"
            )

    return np.asarray(
        X,
        dtype=np.float32
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            IMG_SIZE,
            IMG_SIZE,
            CHANNELS
        )
    )

    # ================================================
    # CNN FEATURE EXTRACTION
    # ================================================

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

    # ================================================
    # TEMPORAL MODEL
    # ================================================

    x = tf.keras.layers.LSTM(
        128,
        return_sequences=False
    )(x)

    x = tf.keras.layers.Dropout(
        0.5
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# MAIN TRAINING
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DEEPFAKE DETECTION TRAINING")
    print("=" * 60)

    # --------------------------------------------
    # Load paths
    # --------------------------------------------

    paths, labels = load_dataset()

    if len(paths) < 10:

        raise ValueError(
            "Not enough processed sequences."
        )

    # --------------------------------------------
    # Split
    # --------------------------------------------

    (
        train_paths,
        test_paths,
        train_labels,
        test_labels
    ) = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    (
        train_paths,
        val_paths,
        train_labels,
        val_labels
    ) = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_labels
    )

    print(
        f"\nTraining samples: "
        f"{len(train_paths)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_paths)}"
    )

    print(
        f"Test samples: "
        f"{len(test_paths)}"
    )

    # --------------------------------------------
    # Load arrays
    # --------------------------------------------

    print("\nLoading training data...")

    X_train = load_sequences(
        train_paths
    )

    y_train = train_labels[
        :len(X_train)
    ]

    print("\nLoading validation data...")

    X_val = load_sequences(
        val_paths
    )

    y_val = val_labels[
        :len(X_val)
    ]

    # --------------------------------------------
    # Model
    # --------------------------------------------

    print("\nBuilding model...")

    model = build_model()

    model.summary()

    # --------------------------------------------
    # Callbacks
    # --------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --------------------------------------------
    # Train
    # --------------------------------------------

    print("\nStarting training...")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------
    # Load best model
    # --------------------------------------------

    if os.path.exists(
        MODEL_PATH
    ):

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

    # --------------------------------------------
    # Test
    # --------------------------------------------

    print("\nEvaluating model...")

    X_test = load_sequences(
        test_paths
    )

    y_test = test_labels[
        :len(X_test)
    ]

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

    with open(
        METRICS_PATH,
        "w"
    ) as file:

        import json

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
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

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nMetrics saved to:"
    )

    print(
        METRICS_PATH
    )

    print("=" * 60)