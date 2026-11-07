"""Pilot: ViT fake-probability separation on a subset of val+test videos (ONNX)."""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_PATH = BASE_DIR / "backend" / "experimental_sequences" / "v2_split.csv"
ONNX_PATH = BASE_DIR / "models" / "vit_deepfake.onnx"
OUT_PATH = BASE_DIR / "backend" / "v2_feature_cache" / "vit_frame_probs_pilot.npz"
PER_SPLIT = 30
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

    rows = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8", newline="")))
    picked = []
    for split in ("validation", "test"):
        for lab in ("0", "1"):
            sel = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == lab][: PER_SPLIT // 2]
            picked.extend(sel)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    probs = np.zeros((len(picked), 20), dtype=np.float32)
    labels = np.array([int(rows[i]["label"]) for i in picked])
    splits = np.array([rows[i]["split"] for i in picked])
    t0 = time.time()
    for k, idx in enumerate(picked):
        seq = np.load(BASE_DIR / rows[idx]["sequence_path"].replace("\\", "/"), allow_pickle=False)
        x = preprocess(seq)
        logits = sess.run(["logits"], {"pixel_values": x})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs[k] = (e / e.sum(axis=1, keepdims=True))[:, FAKE_INDEX]
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{k + 1}/{len(picked)} | {el:.0f}s | ETA {el / (k + 1) * (len(picked) - k - 1):.0f}s", flush=True)

    mean_prob = probs.mean(axis=1)
    from sklearn.metrics import roc_auc_score, accuracy_score

    for split in ("validation", "test"):
        m = splits == split
        if m.sum() == 0:
            continue
        auc = roc_auc_score(labels[m], mean_prob[m])
        acc = accuracy_score(labels[m], (mean_prob[m] >= 0.5).astype(int))
        print(f"{split}: n={m.sum()} AUC={auc:.4f} acc@0.5={acc:.4f} "
              f"real_mean={mean_prob[m][labels[m]==0].mean():.3f} fake_mean={mean_prob[m][labels[m]==1].mean():.3f}")
    np.savez(OUT_PATH, indices=np.array(picked), probs=probs, labels=labels, splits=splits)
    print("saved", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
