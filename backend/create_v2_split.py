
import csv
import re
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    BASE_DIR / "backend" / "experimental_sequences"
    / "experimental_manifest.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "backend" / "experimental_sequences"
    / "v2_split.csv"
)

print("\n" + "=" * 65)
print("DEEPFAKE DETECTOR V2 - DATASET SPLITTING")
print("=" * 65)

# ---------------------------------------------------------
# Load only successfully processed sequences
# ---------------------------------------------------------

with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

rows = [
    row for row in rows
    if row["status"] == "processed"
]

if not rows:
    raise ValueError("No processed sequences found.")

# ---------------------------------------------------------
# Create a shared group identifier.
#
# REAL_10 and FAKE_10 will both belong to GROUP_10.
# This conservatively keeps potentially corresponding
# source videos in the same split.
# ---------------------------------------------------------

labels = []
groups = []

for row in rows:

    class_name = row["class_name"].upper()

    if class_name == "REAL":
        label = 0
    elif class_name == "FAKE":
        label = 1
    else:
        raise ValueError(f"Unknown class: {class_name}")

    match = re.search(r"(\d+)$", row["source_group"])

    if not match:
        raise ValueError(
            f"Could not identify group number: {row['source_group']}"
        )

    group_id = f"GROUP_{match.group(1)}"

    row["label"] = label
    row["v2_group"] = group_id

    labels.append(label)
    groups.append(group_id)

# ---------------------------------------------------------
# Five-fold stratified group split
# Fold 1 = test
# Fold 2 = validation
# Folds 3, 4, 5 = training
# ---------------------------------------------------------

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_assignments = {}

for fold_number, (_, validation_indices) in enumerate(
    splitter.split(
        X=rows,
        y=labels,
        groups=groups
    ),
    start=1
):

    for index in validation_indices:
        fold_assignments[index] = fold_number

for index, row in enumerate(rows):

    fold = fold_assignments[index]

    if fold == 1:
        row["split"] = "test"
    elif fold == 2:
        row["split"] = "validation"
    else:
        row["split"] = "train"

# ---------------------------------------------------------
# Verify group separation
# ---------------------------------------------------------

split_groups = {}

for split_name in ["train", "validation", "test"]:

    split_groups[split_name] = {
        row["v2_group"]
        for row in rows
        if row["split"] == split_name
    }

for first in split_groups:
    for second in split_groups:

        if first >= second:
            continue

        overlap = (
            split_groups[first] & split_groups[second]
        )

        if overlap:
            raise RuntimeError(
                f"Group leakage between {first} and {second}: "
                f"{sorted(overlap)}"
            )

# ---------------------------------------------------------
# Print class distribution
# ---------------------------------------------------------

print("\nSPLIT SUMMARY")
print("-" * 65)

for split_name in ["train", "validation", "test"]:

    selected = [
        row for row in rows
        if row["split"] == split_name
    ]

    counts = Counter(
        row["class_name"].upper()
        for row in selected
    )

    print(
        f"{split_name.upper():12} | "
        f"Total: {len(selected):4} | "
        f"REAL: {counts['REAL']:4} | "
        f"FAKE: {counts['FAKE']:4} | "
        f"Groups: {len(split_groups[split_name]):3}"
    )

print("\nGROUP OVERLAP CHECK")
print("-" * 65)

print("Train / Validation:",
      len(split_groups["train"] & split_groups["validation"]))

print("Train / Test:",
      len(split_groups["train"] & split_groups["test"]))

print("Validation / Test:",
      len(split_groups["validation"] & split_groups["test"]))

# ---------------------------------------------------------
# Save split
# ---------------------------------------------------------

fieldnames = list(rows[0].keys())

with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", OUTPUT_PATH)
print("Total sequences:", len(rows))
print("\nV2 SPLIT COMPLETED SUCCESSFULLY")
