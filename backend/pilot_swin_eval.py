"""Pilot: Swin deepfake-face classifier on stratified val/test videos."""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_PATH = BASE_DIR / "backend" / "experimental_sequences" / "v2_split.csv"
REPO_ID = "romitbarua/autotrain-deepfakeface_only_faces_insightface-94902146221"
OUT_PATH = BASE_DIR / "backend" / "v2_feature_cache" / "swin_frame_probs_pilot.npz"
PER_CLASS = 10
INPUT_SIZE = 224


def main():
    import torch
    from transformers import AutoModelForImageClassification

    rows = list(csv.DictReader(open(SPLIT_PATH, encoding="utf-8", newline="")))
    picked = []
    for split in ("validation", "test"):
        for lab in ("0", "1"):
            picked.extend(
                i for i, r in enumerate(rows) if r["split"] == split and r["label"] == lab
            )
    # interleave so we get a mix early: take first PER_CLASS of each (split,label)
    picked = []
    for split in ("validation", "test"):
        for lab in ("0", "1"):
            sel = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == lab][:PER_CLASS]
            picked.extend(sel)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model = AutoModelForImageClassification.from_pretrained(REPO_ID).eval()
    print("labels:", model.config.id2label)

    all_probs = np.zeros((len(picked), 20, 2), dtype=np.float32)
    labels = np.array([int(rows[i]["label"]) for i in picked])
    t0 = time.time()
    for k, idx in enumerate(picked):
        seq = np.load(BASE_DIR / rows[idx]["sequence_path"].replace("\\", "/"), allow_pickle=False).astype(np.float32)
        frames = []
        for f in seq:
            if f.max() <= 1.5:
                f = f * 255.0
            frames.append(cv2.resize(np.clip(f, 0, 255), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32))
        batch = np.stack(frames).astype(np.float32) / 255.0
        batch = (batch - mean) / std
        batch = batch.transpose(0, 3, 1, 2)
        inputs = torch.from_numpy(batch)
        with torch.no_grad():
            logits = model(pixel_values=inputs).logits
            all_probs[k] = torch.softmax(logits, dim=-1).numpy()
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{k + 1}/{len(picked)} | {el:.0f}s | ETA {el / (k + 1) * (len(picked) - k - 1):.0f}s", flush=True)

    np.savez(OUT_PATH, indices=np.array(picked), probs=all_probs, labels=labels)

    # report: prob of class index for 'fake-ish' — print both class means per label
    for cls in (0, 1):
        mp = all_probs[:, :, cls].mean(axis=1)
        print(f"class {cls} ({model.config.id2label.get(str(cls), cls)}): "
              f"real_mean={mp[labels==0].mean():.3f} fake_mean={mp[labels==1].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
