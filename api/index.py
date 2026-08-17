import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import tempfile
import time
import traceback

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

SEQUENCE_LENGTH = 20

IMG_SIZE = 128

MAX_FILE_SIZE = 100 * 1024 * 1024


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

CORS(app)

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_FILE_SIZE


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("DEEPFAKE DETECTION BACKEND")
print("=" * 60)

print(
    "\nLoading model..."
)

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n"
        "\nRun train.py first."
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully."
)

print(
    "Model input shape:",
    model.input_shape
)


# ============================================================
# LOAD METRICS
# ============================================================

def get_metrics():

    if not os.path.exists(
        METRICS_PATH
    ):

        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "test_samples": None
        }

    try:

        with open(
            METRICS_PATH,
            "r"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "test_samples": None
        }


# ============================================================
# PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    # OpenCV BGR -> RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = cv2.resize(
        frame,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    frame = frame.astype(
        np.float32
    )

    frame /= 255.0

    return frame


# ============================================================
# EXTRACT VIDEO SEQUENCE
# ============================================================

def extract_video_sequence(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise ValueError(
            "Unable to open video."
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise ValueError(
            "Video has no readable frames."
        )

    # Select 20 evenly spaced frames
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if not ret:

            continue

        processed = preprocess_frame(
            frame
        )

        frames.append(
            processed
        )

    cap.release()

    if len(frames) == 0:

        raise ValueError(
            "Could not extract frames."
        )

    # Pad missing frames
    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            frames[-1].copy()
        )

    frames = frames[
        :SEQUENCE_LENGTH
    ]

    return np.asarray(
        frames,
        dtype=np.float32
    )


# ============================================================
# PREDICT
# ============================================================

def predict_sequence(
    sequence
):

    input_data = np.expand_dims(
        sequence,
        axis=0
    )

    prediction = model.predict(
        input_data,
        verbose=0
    )

    # Model has sigmoid output
    fake_probability = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    fake_probability = max(
        0.0,
        min(
            1.0,
            fake_probability
        )
    )

    real_probability = (
        1.0 -
        fake_probability
    )

    is_fake = (
        fake_probability >= 0.5
    )

    confidence = (
        fake_probability
        if is_fake
        else real_probability
    )

    return {

        "is_fake":
            bool(is_fake),

        "label":
            "FAKE"
            if is_fake
            else "REAL",

        "confidence":
            round(
                confidence,
                4
            ),

        "fake_probability":
            round(
                fake_probability,
                4
            ),

        "real_probability":
            round(
                real_probability,
                4
            )
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Unable to read image."
        )

    frame = preprocess_frame(
        image
    )

    # Repeat image 20 times
    sequence = np.repeat(
        frame[np.newaxis, ...],
        SEQUENCE_LENGTH,
        axis=0
    )

    return predict_sequence(
        sequence
    )


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(
    video_path
):

    sequence = extract_video_sequence(
        video_path
    )

    return predict_sequence(
        sequence
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "success": True,

        "name":
            "Deepfake Detection API",

        "status":
            "running",

        "model":
            "deepfake_model.h5",

        "input":
            "image or video",

        "sequence_length":
            SEQUENCE_LENGTH,

        "image_size":
            IMG_SIZE
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success": True,

        "status":
            "healthy",

        "model_loaded":
            True
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Accept:
    #
    # file
    # image
    # video
    # --------------------------------------------------------

    uploaded_file = None

    if "file" in request.files:

        uploaded_file = request.files[
            "file"
        ]

    elif "image" in request.files:

        uploaded_file = request.files[
            "image"
        ]

    elif "video" in request.files:

        uploaded_file = request.files[
            "video"
        ]

    if uploaded_file is None:

        return jsonify({

            "success": False,

            "error":
                "No file uploaded."
        }), 400

    if uploaded_file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No file selected."
        }), 400


    filename = uploaded_file.filename

    extension = os.path.splitext(
        filename
    )[1].lower()


    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }


    if extension in image_extensions:

        file_type = "image"

    elif extension in video_extensions:

        file_type = "video"

    else:

        return jsonify({

            "success": False,

            "error":
                "Unsupported file type."
        }), 400


    temp_path = None


    try:

        # ----------------------------------------------------
        # Temporary file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            uploaded_file.save(
                temp.name
            )

            temp_path = temp.name


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        start_time = time.time()

        if file_type == "image":

            result = predict_image(
                temp_path
            )

        else:

            result = predict_video(
                temp_path
            )


        processing_time = (
            time.time()
            -
            start_time
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        result.update({

            "success": True,

            "file_name":
                filename,

            "file_type":
                file_type,

            "processing_time":
                round(
                    processing_time,
                    3
                )
        })


        return jsonify(
            result
        ), 200


    except Exception as e:

        print(
            "\nPrediction error:"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
                str(e)
        }), 500


    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# METRICS
# ============================================================

@app.route(
    "/metrics",
    methods=["GET"]
)
def metrics():

    return jsonify({

        "success": True,

        "metrics":
            get_metrics()
    })


# ============================================================
# MODEL INFO
# ============================================================

@app.route(
    "/model-info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "success": True,

        "framework":
            "TensorFlow / Keras",

        "model":
            "CNN + LSTM",

        "input_shape":
            str(
                model.input_shape
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "image_size":
            IMG_SIZE,

        "classes": {

            "0":
                "REAL",

            "1":
                "FAKE"
        },

        "decision_threshold":
            0.5
    })


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "File is too large. Maximum size is 100 MB."
    }), 413


# ============================================================
# NOT FOUND
# ============================================================

@app.errorhandler(
    404
)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
            "API endpoint not found."
    }), 404


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SERVER STARTING")
    print("=" * 60)

    print(
        "URL: http://localhost:5000"
    )

    print(
        "Prediction: "
        "POST /predict"
    )

    print(
        "Metrics: "
        "GET /metrics"
    )

    print(
        "Health: "
        "GET /health"
    )

    print(
        "Model info: "
        "GET /model-info"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )