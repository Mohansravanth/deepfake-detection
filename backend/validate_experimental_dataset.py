
import csv
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"
MANIFEST_PATH = DATA_DIR / "experimental_manifest.csv"

REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"

EXPECTED_SHAPE = (20, 128, 128, 3)

print("\n" + "=" * 65)
print("EXPERIMENTAL DATASET VALIDATION")
print("=" * 65)

# --------------------------------------------------
# 1. Read manifest
# --------------------------------------------------

with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("\nMANIFEST SUMMARY")
print("-" * 65)

print("Total manifest records:", len(rows))

status_counts = Counter(row["status"] for row in rows)
class_counts = Counter(row["class_name"] for row in rows)
label_counts = Counter(row["label"] for row in rows)

print("Status distribution:", dict(status_counts))
print("Class distribution:", dict(class_counts))
print("Label distribution:", dict(label_counts))

# --------------------------------------------------
# 2. Validate source groups
# --------------------------------------------------

real_groups = {
    row["source_group"]
    for row in rows
    if row["class_name"].upper() == "REAL"
    and row["status"].lower() == "processed"
}

fake_groups = {
    row["source_group"]
    for row in rows
    if row["class_name"].upper() == "FAKE"
    and row["status"].lower() == "processed"
}

print("\nSOURCE GROUP VALIDATION")
print("-" * 65)

print("Distinct real source groups:", len(real_groups))
print("Distinct fake source groups:", len(fake_groups))

print("Real groups:", sorted(real_groups))
print("Fake groups:", sorted(fake_groups))

# --------------------------------------------------
# 3. Validate saved NumPy sequences
# --------------------------------------------------

results = {}

for class_name, folder in [
    ("REAL", REAL_DIR),
    ("FAKE", FAKE_DIR)
]:

    files = sorted(folder.glob("*.npy"))

    shape_counts = Counter()
    dtype_counts = Counter()

    invalid_shape = []
    invalid_dtype = []
    invalid_range = []
    invalid_values = []

    minimum_values = []
    maximum_values = []

    total_size = 0

    print(f"\n{class_name} SEQUENCE VALIDATION")
    print("-" * 65)

    for path in files:

        try:
            sequence = np.load(path, allow_pickle=False)

            shape_counts[str(sequence.shape)] += 1
            dtype_counts[str(sequence.dtype)] += 1

            total_size += path.stat().st_size

            if sequence.shape != EXPECTED_SHAPE:
                invalid_shape.append(path.name)
                continue

            if sequence.dtype != np.float32:
                invalid_dtype.append(path.name)

            if not np.isfinite(sequence).all():
                invalid_values.append(path.name)
                continue

            minimum = float(sequence.min())
            maximum = float(sequence.max())

            minimum_values.append(minimum)
            maximum_values.append(maximum)

            if minimum < 0.0 or maximum > 1.0:
                invalid_range.append(path.name)

        except Exception as error:
            invalid_values.append(
                f"{path.name}: {error}"
            )

    results[class_name] = {
        "count": len(files),
        "invalid_shape": len(invalid_shape),
        "invalid_dtype": len(invalid_dtype),
        "invalid_range": len(invalid_range),
        "invalid_values": len(invalid_values)
    }

    print("Sequence files:", len(files))
    print("Shape distribution:", dict(shape_counts))
    print("Data type distribution:", dict(dtype_counts))

    print("Invalid shapes:", len(invalid_shape))
    print("Invalid data types:", len(invalid_dtype))
    print("Invalid pixel ranges:", len(invalid_range))
    print("Invalid/non-finite sequences:", len(invalid_values))

    if minimum_values:
        print("Minimum pixel value:", min(minimum_values))

    if maximum_values:
        print("Maximum pixel value:", max(maximum_values))

    print(
        "Total storage:",
        round(total_size / (1024 ** 3), 3),
        "GB"
    )

# --------------------------------------------------
# 4. Final validation
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL VALIDATION RESULT")
print("=" * 65)

expected_counts = {
    "REAL": 456,
    "FAKE": 454
}

all_valid = True

for class_name in ["REAL", "FAKE"]:

    result = results[class_name]

    print(f"\n{class_name}:")

    for key, value in result.items():
        print(f"{key}: {value}")

    if result["count"] != expected_counts[class_name]:
        all_valid = False

    if any(
        result[key] != 0
        for key in [
            "invalid_shape",
            "invalid_dtype",
            "invalid_range",
            "invalid_values"
        ]
    ):
        all_valid = False

if len(real_groups) != 53 or len(fake_groups) != 53:
    all_valid = False

if all_valid:
    print("\nRESULT: ALL SEQUENCE CHECKS PASSED")
else:
    print("\nRESULT: VALIDATION ISSUES FOUND")

print("=" * 65)
