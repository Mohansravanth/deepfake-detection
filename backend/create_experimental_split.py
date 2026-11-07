
import csv
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"
MANIFEST_PATH = DATA_DIR / "experimental_manifest.csv"
OUTPUT_PATH = DATA_DIR / "experimental_split.csv"

RANDOM_STATE = 42
N_SPLITS = 5

# --------------------------------------------------
# 1. Load only successfully processed sequences
# --------------------------------------------------

with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
    all_rows = list(csv.DictReader(file))

rows = [
    row for row in all_rows
    if row["status"].lower() == "processed"
]

labels = np.array(
    [int(row["label"]) for row in rows]
)

groups = np.array(
    [row["source_group"] for row in rows]
)

indices = np.arange(len(rows))

print("\n" + "=" * 65)
print("GROUP-AWARE DATASET SPLITTING")
print("=" * 65)

print("Total usable sequences:", len(rows))
print("Real sequences:", int(np.sum(labels == 0)))
print("Fake sequences:", int(np.sum(labels == 1)))
print("Unique source groups:", len(set(groups)))

# --------------------------------------------------
# 2. Create five stratified group folds
# --------------------------------------------------

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = []

for fold_number, (train_indices, test_indices) in enumerate(
    splitter.split(indices, labels, groups)
):
    folds.append(test_indices)

    fold_labels = labels[test_indices]

    print(f"\nFold {fold_number + 1}")
    print("-" * 40)
    print("Samples:", len(test_indices))
    print("Real:", int(np.sum(fold_labels == 0)))
    print("Fake:", int(np.sum(fold_labels == 1)))
    print("Groups:", len(set(groups[test_indices])))

# --------------------------------------------------
# 3. Assign train, validation and test sets
# --------------------------------------------------

test_indices = folds[0]
validation_indices = folds[1]

train_indices = np.concatenate(folds[2:])

assignments = {}

for index in train_indices:
    assignments[int(index)] = "train"

for index in validation_indices:
    assignments[int(index)] = "validation"

for index in test_indices:
    assignments[int(index)] = "test"

# --------------------------------------------------
# 4. Verify group separation
# --------------------------------------------------

train_groups = set(groups[train_indices])
validation_groups = set(groups[validation_indices])
test_groups = set(groups[test_indices])

assert train_groups.isdisjoint(validation_groups)
assert train_groups.isdisjoint(test_groups)
assert validation_groups.isdisjoint(test_groups)

assert len(assignments) == len(rows)

# --------------------------------------------------
# 5. Print final split statistics
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DATASET SPLIT")
print("=" * 65)

for split_name, split_indices in [
    ("TRAIN", train_indices),
    ("VALIDATION", validation_indices),
    ("TEST", test_indices)
]:

    split_labels = labels[split_indices]
    split_groups = groups[split_indices]

    print(f"\n{split_name}")
    print("-" * 40)
    print("Total samples:", len(split_indices))
    print("Real samples:", int(np.sum(split_labels == 0)))
    print("Fake samples:", int(np.sum(split_labels == 1)))
    print("Source groups:", len(set(split_groups)))
    print("Percentage:", round(len(split_indices) / len(rows) * 100, 2))

# --------------------------------------------------
# 6. Save split manifest
# --------------------------------------------------

fieldnames = list(rows[0].keys()) + ["split"]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for index, row in enumerate(rows):
        output_row = dict(row)
        output_row["split"] = assignments[index]
        writer.writerow(output_row)

# --------------------------------------------------
# 7. Final checks
# --------------------------------------------------

assert len(train_indices) + len(validation_indices) + len(test_indices) == len(rows)

print("\n" + "=" * 65)
print("SPLIT VALIDATION")
print("=" * 65)

print("Train/validation overlap:", len(train_groups & validation_groups))
print("Train/test overlap:", len(train_groups & test_groups))
print("Validation/test overlap:", len(validation_groups & test_groups))

print("\nSaved split manifest:")
print(OUTPUT_PATH)

print("\nRESULT: GROUP-AWARE SPLIT COMPLETED")
print("=" * 65)
