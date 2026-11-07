"""Compute per-frame deepfake probabilities with the ONNX ViT model.

Runs models/vit_deepfake.onnx (exported from dima806/deepfake_vs_real_image_detection)
over every frame of every cached sequence in v2_split.csv and stores the
result as an (N, SEQUENCE_LENGTH) float32 array for offline ensembling.
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_PATH = BASE_DIR / "backend" / "experimental_sequences" / "v2_split.csv"
ONNX_PATH = BASE_DIR / "models" / "vit_deepfake.onnx"
OUT_DIR = BASE_DIR / "backend" / "v2_feature_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBS_PATH = OUT_DIR / "vit_frame_probs.npy"
META_PATH = OUT_DIR / "vit_frame_probs.json"

INPUT_SIZE = 224
FAKE_INDEX = 1


def preprocess(sequence):
    frames = np.zeros((len(sequence), 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    for i, f in enumerate(sequence):
        f = f.astype(np.float32)
        if f.max() <= 1.5:
            f = f * 255.0
        r = cv2.resize(np.clip(f, 0, 255), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA) / 255.0
        r = (r - 0.5) / 0.5
        frames[i] = r.transpose(2, 0, 1)
    return frames


def main():
    import onnxruntime as ort

    rows = list(csv.DictReader(open(SPLIT_PATH, "r", encoding="utf-8", newline="")))
    n = len(rows)
    seq_len = 20

    if PROBS_PATH.exists():
        cached = np.load(PROBS_PATH)
        meta = json.load(open(META_PATH)) if META_PATH.exists() else {}
        if cached.shape == (n, seq_len) and meta.get("onnx") == "vit_deepfake.onnx":
            print(f"Cache valid: {PROBS_PATH} {cached.shape}")
            return 0

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    probs = np.zeros((n, seq_len), dtype=np.float32)
    t0 = time.time()
    for start in range(n):
        row = rows[start]
        seq_path = BASE_DIR / row["sequence_path"].replace("\\", "/")
        sequence = np.load(seq_path, allow_pickle=False)
        if sequence.ndim != 4:
            raise ValueError(f"Bad sequence {seq_path}: {sequence.shape}")
        x = preprocess(sequence)
        logits = session.run(["logits"], {"pixel_values": x})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs[start] = (e / e.sum(axis=1, keepdims=True))[:, FAKE_INDEX]
        done = start + 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            print(f"{done}/{n} videos | {elapsed:.0f}s | ETA {elapsed / done * (n - done):.0f}s", flush=True)

    np.save(PROBS_PATH, probs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "onnx": "vit_deepfake.onnx",
                "source": "dima806/deepfake_vs_real_image_detection",
                "input_size": INPUT_SIZE,
                "sequence_length": seq_len,
                "shape": list(probs.shape),
            },
            f,
            indent=2,
        )
    print(f"Saved {PROBS_PATH} {probs.shape} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
