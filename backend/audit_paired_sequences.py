import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "backend" / "experimental_sequences"

REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"

print("\n" + "=" * 65)
print("REAL AND FAKE PAIRED SEQUENCE COMPARISON")
print("=" * 65)

print("\nComparing corresponding numbered source groups.")
print("This is an exploratory comparison, not proof of pairing.\n")

results = []

for group_number in range(1, 54):

    real_files = list(
        REAL_DIR.glob(f"*real_v{group_number}_aug_0*.npy")
    )

    fake_files = list(
        FAKE_DIR.glob(f"*fake_vs{group_number}_aug_0*.npy")
    )

    if not real_files or not fake_files:
        continue

    real = np.load(real_files[0], allow_pickle=False)
    fake = np.load(fake_files[0], allow_pickle=False)

    if real.shape != fake.shape:
        print(
            f"Group {group_number}: SHAPE MISMATCH "
            f"{real.shape} vs {fake.shape}"
        )
        continue

    difference = np.abs(real - fake)

    mean_difference = float(np.mean(difference))

    changed_percentage = float(
        np.mean(np.any(difference > 0.05, axis=-1)) * 100
    )

    frame_differences = []

    for frame_index in range(20):

        frame_diff = np.mean(
            np.abs(real[frame_index] - fake[frame_index])
        )

        frame_differences.append(float(frame_diff))

    results.append(
        (group_number, mean_difference, changed_percentage)
    )

    print(
        f"Group {group_number:02d} | "
        f"Mean difference: {mean_difference:.6f} | "
        f"Changed pixels: {changed_percentage:.2f}%"
    )

print("\n" + "=" * 65)
print("COMPARISON SUMMARY")
print("=" * 65)

print("Comparable groups:", len(results))

if results:

    differences = [item[1] for item in results]

    print(
        "Average mean difference:",
        round(float(np.mean(differences)), 6)
    )

    print(
        "Minimum mean difference:",
        round(float(np.min(differences)), 6)
    )

    print(
        "Maximum mean difference:",
        round(float(np.max(differences)), 6)
    )

print("\nCOMPARISON COMPLETED")