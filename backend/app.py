import json
import os
import sys
import tempfile
import time
import traceback
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from forensic_features import extract_forensic_vector, extract_media_sequence

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMPROVED_MODEL_PATH = os.path.join(MODEL_DIR, "improved_detector.joblib")
V2_MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model_v2.keras")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.h5")
METRICS_PATHS = [
    os.path.join(MODEL_DIR, "metrics.json"),
    os.path.join(BACKEND_DIR, "data", "metrics.json"),
]
ENSEMBLE_CONFIG_PATH = os.path.join(MODEL_DIR, "ensemble_config.json")
FRAME_CNN_PATH = os.path.join(MODEL_DIR, "frame_cnn.keras")
VIT_ONNX_PATH = os.path.join(MODEL_DIR, "vit_deepfake.onnx")
VIT_TEMPORAL_HEAD_PATH = os.path.join(BACKEND_DIR, "v2_feature_cache", "vit_temporal_head.joblib")
VIT_FRAME_GBM_PATH = os.path.join(BACKEND_DIR, "v2_feature_cache", "vit_frame_gbm.joblib")
VIT_INPUT_SIZE = 224
VIT_FAKE_INDEX = 1
FRONTEND_BUILD = os.path.join(BASE_DIR, "frontend", "build")
SEQUENCE_LENGTH = 20
IMG_SIZE = 128
MAX_FILE_SIZE = 100 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__, static_folder=FRONTEND_BUILD, static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

improved_bundle = None
tf_model = None
tf_threshold = 0.5
vit_session = None
vit_heads = {}
frame_cnn = None
ensemble_config = None


def load_ensemble_config():
    global ensemble_config
    if ensemble_config is not None:
        return ensemble_config
    if os.path.exists(ENSEMBLE_CONFIG_PATH):
        try:
            with open(ENSEMBLE_CONFIG_PATH, "r", encoding="utf-8") as file:
                ensemble_config = json.load(file)
        except Exception as exc:
            print("Failed to load ensemble config:", exc)
            ensemble_config = {}
    else:
        ensemble_config = {}
    return ensemble_config


def load_vit_model():
    """Load the ONNX-exported ViT per-frame deepfake classifier."""
    global vit_session
    if vit_session is not None or os.environ.get("LOAD_VIT_MODEL", "1").strip() == "0":
        return vit_session
    if not os.path.exists(VIT_ONNX_PATH):
        print("ViT ONNX model not found:", VIT_ONNX_PATH)
        return None
    try:
        import onnxruntime as ort

        vit_session = ort.InferenceSession(VIT_ONNX_PATH, providers=["CPUExecutionProvider"])
        print("Loaded ViT frame detector (ONNX):", VIT_ONNX_PATH)
    except Exception as exc:
        print("ViT frame detector not available:", exc)
        vit_session = None
    return vit_session


def load_frame_cnn():
    """Load the frame-level CNN trained on this project's face crops."""
    global frame_cnn
    if frame_cnn is not None or os.environ.get("LOAD_FRAME_CNN", "1").strip() == "0":
        return frame_cnn
    if not os.path.exists(FRAME_CNN_PATH):
        return None
    try:
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        frame_cnn = tf.keras.models.load_model(FRAME_CNN_PATH, compile=False)
        print("Loaded frame CNN:", FRAME_CNN_PATH)
    except Exception as exc:
        print("Frame CNN not available:", exc)
        frame_cnn = None
    return frame_cnn


def frame_cnn_probability(sequence):
    """Mean fake probability over stride-2 frames, matching training."""
    model = load_frame_cnn()
    if model is None:
        return None
    frames = sequence[::2]
    batch = np.stack(frames).astype(np.float32)
    if batch.max() <= 1.5:
        batch = batch * 255.0
    preds = model.predict(np.clip(batch, 0, 255), verbose=0).reshape(-1)
    return float(np.mean(preds))


def load_vit_heads():
    global vit_heads
    if vit_heads:
        return vit_heads
    for name, path in (("vit_head", VIT_TEMPORAL_HEAD_PATH), ("vit_gbm", VIT_FRAME_GBM_PATH)):
        if os.path.exists(path):
            try:
                vit_heads[name] = joblib.load(path)
            except Exception as exc:
                print(f"Failed to load {name}:", exc)
    return vit_heads


def load_improved_model():
    global improved_bundle
    if improved_bundle is not None:
        return improved_bundle
    if os.path.exists(IMPROVED_MODEL_PATH):
        improved_bundle = joblib.load(IMPROVED_MODEL_PATH)
        print("Loaded improved detector:", IMPROVED_MODEL_PATH)
    return improved_bundle


def load_tf_model():
    global tf_model, tf_threshold
    if tf_model is not None:
        return tf_model
    try:
        import tensorflow as tf

        for path, default_threshold in (
            (V2_MODEL_PATH, 0.37),
            (LEGACY_MODEL_PATH, 0.5),
        ):
            if os.path.exists(path):
                tf_model = tf.keras.models.load_model(path, compile=False)
                tf_threshold = default_threshold
                print("Loaded neural model:", path)
                break
    except Exception as exc:
        print("Neural model not available:", exc)
        tf_model = None
    return tf_model


def get_metrics():
    for path in METRICS_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            continue
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "test_samples": None,
    }


def vit_frame_probabilities(sequence):
    """Run the ViT detector on each face crop; returns (T,) fake probs or None."""
    session = load_vit_model()
    if session is None:
        return None

    n = len(sequence)
    batch = np.zeros((n, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE), dtype=np.float32)
    for i, frame in enumerate(sequence):
        rgb = np.asarray(frame, dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0)
        rgb = cv2.resize(rgb, (VIT_INPUT_SIZE, VIT_INPUT_SIZE), interpolation=cv2.INTER_AREA) / 255.0
        rgb = (rgb - 0.5) / 0.5
        batch[i] = rgb.transpose(2, 0, 1)

    logits = session.run(["logits"], {"pixel_values": batch})[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (exp / exp.sum(axis=1, keepdims=True))[:, VIT_FAKE_INDEX]
    return probs.astype(np.float32)


def vit_stats(probs):
    t = np.arange(probs.shape[0], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    slope = float((probs * t).sum() / (t * t).sum())
    return np.array(
        [
            probs.mean(),
            probs.std(),
            probs.min(),
            probs.max(),
            float(np.median(probs)),
            float(np.mean(probs > 0.5)),
            slope,
            float(probs[-1] - probs[0]),
        ],
        dtype=np.float32,
    ).reshape(1, -1)


def compute_signals(sequence, face_coverage):
    """Return dict of signal_name -> fake probability in [0,1]."""
    signals = {}
    wanted = set((load_ensemble_config().get("signals") or {}).keys())

    def need(name):
        return not wanted or name in wanted

    forensic_vector = extract_forensic_vector(sequence, face_coverage=face_coverage).reshape(1, -1)
    bundle = load_improved_model()
    if need("forensic") and bundle is not None:
        signals["forensic"] = float(bundle["forensic_model"].predict_proba(forensic_vector)[0, 1])

    if need("framecnn"):
        frame_prob = frame_cnn_probability(sequence)
        if frame_prob is not None:
            signals["framecnn"] = frame_prob

    if need("vit_mean") or need("vit_head") or need("vit_gbm"):
        frame_probs = vit_frame_probabilities(sequence)
        if frame_probs is not None:
            if need("vit_mean"):
                signals["vit_mean"] = float(frame_probs.mean())
            heads = load_vit_heads()
            if need("vit_head") and "vit_head" in heads:
                signals["vit_head"] = float(heads["vit_head"].predict_proba(vit_stats(frame_probs))[0, 1])
            if need("vit_gbm") and "vit_gbm" in heads:
                signals["vit_gbm"] = float(heads["vit_gbm"].predict_proba(frame_probs.reshape(1, -1))[0, 1])

    if need("tf_seq") and os.environ.get("LOAD_TF_MODEL", "").strip() == "1":
        model = load_tf_model()
        if model is not None:
            preds = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
            signals["tf_seq"] = float(np.asarray(preds).reshape(-1)[0])

    return signals


def blend_signals(signals):
    """Weighted blend defined by models/ensemble_config.json (tuned on validation)."""
    config = load_ensemble_config()
    weights = config.get("signals") or {}
    threshold = float(config.get("threshold", 0.5))

    if not weights:
        # Fallback: forensic-only if available, else best single signal.
        if "forensic" in signals:
            weights = {"forensic": 1.0}
        elif signals:
            weights = {next(iter(signals)): 1.0}
        else:
            return 0.5, 0.5

    if not signals:
        return 0.5, 0.5

    available = {name: prob for name, prob in signals.items() if name in weights}
    if not available:
        # Config references signals we could not compute; use whatever exists.
        available = signals
        total = sum(available.values()) or 1.0
        fake_probability = sum(probs * w for probs, w in zip(available.values(), [v / total for v in available.values()]))
        return float(np.clip(fake_probability, 0.0, 1.0)), threshold

    weight_sum = sum(weights[name] for name in available)
    if weight_sum <= 0:
        weight_sum = 1.0
    fake_probability = sum(weights[name] * prob for name, prob in available.items()) / weight_sum
    return float(np.clip(fake_probability, 0.0, 1.0)), threshold


def predict_with_models(sequence, face_coverage):
    signals = compute_signals(sequence, face_coverage)

    forensic_prob = signals.get("forensic")
    neural_prob = signals.get("tf_seq")

    if forensic_prob is None and neural_prob is None:
        # Last-resort legacy blend.
        if signals:
            fake_probability, threshold = blend_signals(signals)
        else:
            fake_probability, threshold = 0.5, 0.5
    else:
        fake_probability, threshold = blend_signals(signals)

    fake_probability = float(np.clip(fake_probability, 0.0, 1.0))
    is_fake = fake_probability >= threshold
    real_probability = 1.0 - fake_probability
    confidence = fake_probability if is_fake else real_probability
    return {
        "is_fake": bool(is_fake),
        "label": "FAKE" if is_fake else "REAL",
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "decision_threshold": round(threshold, 4),
        "face_detected": bool(face_coverage > 0.01),
        "signals": {name: round(float(prob), 4) for name, prob in signals.items()},
    }


def predict_media(path):
    sequence, face_coverage = extract_media_sequence(path, SEQUENCE_LENGTH)
    return predict_with_models(sequence, face_coverage)


def allowed_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image", extension
    if extension in VIDEO_EXTENSIONS:
        return "video", extension
    return None, extension


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "improved_model": os.path.exists(IMPROVED_MODEL_PATH),
            "neural_model_loaded": tf_model is not None,
            "vit_model_loaded": vit_session is not None,
            "frame_cnn_loaded": frame_cnn is not None,
            "ensemble_config": bool(load_ensemble_config().get("signals")),
        }
    )


@app.route("/api", methods=["GET"])
def api_home():
    return jsonify(
        {
            "success": True,
            "name": "Deepfake Detection API",
            "status": "running",
            "input": "image or video",
            "sequence_length": SEQUENCE_LENGTH,
            "image_size": IMG_SIZE,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = (
        request.files.get("file")
        or request.files.get("image")
        or request.files.get("video")
    )
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    filename = uploaded_file.filename
    file_type, extension = allowed_file(filename)
    if file_type is None:
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            uploaded_file.save(temp.name)
            temp_path = temp.name
        start = time.time()
        result = predict_media(temp_path)
        result.update(
            {
                "success": True,
                "file_name": filename,
                "file_type": file_type,
                "processing_time": round(time.time() - start, 3),
            }
        )
        return jsonify(result), 200
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/add_video_auto", methods=["POST"])
def add_video_auto():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"success": False, "error": "Please upload a video file."}), 400

    extension = os.path.splitext(uploaded_file.filename)[1].lower()
    if extension not in VIDEO_EXTENSIONS:
        return jsonify({"success": False, "error": "Unsupported video format."}), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            uploaded_file.save(temp.name)
            temp_path = temp.name
        sequence, face_coverage = extract_media_sequence(temp_path)
        prediction = predict_with_models(sequence, face_coverage)
        confirmed = request.form.get("confirmed", "false").lower() == "true"
        if not confirmed:
            return jsonify({"success": True, "needs_confirmation": True, **prediction}), 200

        label = "fake" if prediction["is_fake"] else "real"
        dataset_dir = os.environ.get(
            "DATASET_DIR",
            os.path.join(BACKEND_DIR, "processed_sequences"),
        )
        label_dir = os.path.join(dataset_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        sample_name = f"{uuid.uuid4().hex}.npy"
        np.save(
            os.path.join(label_dir, sample_name),
            np.clip(sequence * 255.0, 0, 255).astype(np.uint8),
        )
        return jsonify(
            {
                "success": True,
                "needs_confirmation": False,
                "message": f"Video sample saved to the {label.upper()} training dataset.",
                "label": label.upper(),
                "file_name": sample_name,
            }
        ), 200
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"success": False, "error": "Please enter a question."}), 400
    if len(query) > 1000:
        return jsonify({"success": False, "error": "Question must be 1000 characters or fewer."}), 400

    question = query.lower()
    knowledge = [
        (
            ("what is a deepfake", "define deepfake", "deepfake mean"),
            "A deepfake is synthetic or manipulated media where AI makes a person appear to say or do something they did not.",
        ),
        (
            ("how detect", "detect deepfake", "how does detection"),
            "This site crops faces, samples 20 frames, and scores forensic artifacts plus a learned real/fake classifier. The result is a REAL or FAKE probability, not a legal proof.",
        ),
        (
            ("accuracy", "confidence", "reliable"),
            "Confidence is the model's score for the predicted class. It can still be wrong, especially on compressed social-media video or faces the model has not seen.",
        ),
        (
            ("hello", "hey", "good morning"),
            "Hello. Upload a video or image on the Analyze page, or ask about deepfakes and this detector.",
        ),
    ]
    answer = "I can explain deepfakes, this detector, and how to read REAL vs FAKE scores. Try asking how detection works."
    for keywords, response in knowledge:
        if any(keyword in question for keyword in keywords):
            answer = response
            break
    return jsonify({"success": True, "response": answer, "source": "knowledge-base"}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({"success": True, "metrics": get_metrics()})


@app.route("/model-info", methods=["GET"])
def model_info():
    metrics_data = get_metrics()
    return jsonify(
        {
            "success": True,
            "framework": "OpenCV forensics + scikit-learn ensemble",
            "model": metrics_data.get("model", "improved detector"),
            "sequence_length": SEQUENCE_LENGTH,
            "image_size": IMG_SIZE,
            "classes": {"0": "REAL", "1": "FAKE"},
            "decision_threshold": metrics_data.get("threshold", 0.5),
        }
    )


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File is too large. Maximum size is 100 MB."}), 413


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith(("predict", "health", "metrics", "model-info", "chat", "add_video_auto", "api")):
        return jsonify({"success": False, "error": "API endpoint not found."}), 404
    if os.path.isdir(FRONTEND_BUILD):
        target = os.path.join(FRONTEND_BUILD, path)
        if path and os.path.isfile(target):
            return send_from_directory(FRONTEND_BUILD, path)
        index_path = os.path.join(FRONTEND_BUILD, "index.html")
        if os.path.isfile(index_path):
            return send_from_directory(FRONTEND_BUILD, "index.html")
    return jsonify(
        {
            "success": True,
            "name": "Deepfake Detection API",
            "hint": "Start the React app with npm start, or run npm run build to serve the website from Flask.",
        }
    )


if __name__ == "__main__":
    sys.path.insert(0, BACKEND_DIR)
    load_improved_model()
    config_signals = set((load_ensemble_config().get("signals") or {}).keys())
    if any(name.startswith("vit_") for name in config_signals) or os.environ.get("LOAD_VIT_MODEL", "0").strip() == "1":
        load_vit_model()
        load_vit_heads()
    load_frame_cnn()
    print("Website API: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
