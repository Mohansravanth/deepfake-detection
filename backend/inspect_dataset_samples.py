
import csv
import cv2
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"
MANIFEST_PATH = DATA_DIR / "experimental_manifest.csv"

OUTPUT_DIR = BASE_DIR / "backend" / "dataset_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 5

print("\n" + "=" * 65)
print("DATASET VISUAL INSPECTION")
print("=" * 65)

# --------------------------------------------------
# Load manifest
# --------------------------------------------------

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    rows = list(csv.DictReader(file))

rows = [
    row for row in rows
    if row["status"] == "processed"
]

# --------------------------------------------------
# Select independent source groups
# --------------------------------------------------

for class_name in ["REAL", "FAKE"]:

    class_rows = [
        row for row in rows
        if row["class_name"] == class_name
    ]

    groups = {}

    for row in class_rows:

        group = row["source_group"]

        if group not in groups:
            groups[group] = row

    selected_groups = sorted(groups.keys())[:SAMPLES_PER_CLASS]

    print(f"\n{class_name} SAMPLE INSPECTION")
    print("-" * 65)

    for group in selected_groups:

        row = groups[group]

        relative_path = row["sequence_path"].replace(
            "\\", "/"
        )

        sequence_path = BASE_DIR / relative_path

        sequence = np.load(
            sequence_path,
            allow_pickle=False
        )

        print(
            group,
            "|",
            row["filename"],
            "|",
            "Shape:",
            sequence.shape
        )

        # Select 5 frames across the sequence
        frame_indices = [0, 4, 9, 14, 19]

        frames = []

        for frame_index in frame_indices:

            frame = sequence[frame_index]

            frame = np.clip(
                frame * 255.0,
                0,
                255
            ).astype(np.uint8)

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            frames.append(frame)

        # Combine frames horizontally
        contact_sheet = np.hstack(frames)

        output_path = (
            OUTPUT_DIR
            / f"{class_name}_{group}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            contact_sheet
        )

        print("Saved:", output_path)

print("\n" + "=" * 65)
print("VISUAL INSPECTION COMPLETED")
print("Output directory:", OUTPUT_DIR)
print("=" * 65)
