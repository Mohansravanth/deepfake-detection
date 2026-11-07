import os
from collections import Counter

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASETS = {
    "REAL": os.path.join(BASE_DIR, "SDFVD2.0_real"),
    "FAKE": os.path.join(BASE_DIR, "SDFVD2.0_fake")
}

EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

for category, folder in DATASETS.items():

    print("\n" + "=" * 70)
    print(category, "AUGMENTATION ANALYSIS")
    print("=" * 70)

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS)
    )

    groups = Counter()
    unmatched = []

    for filename in files:

        stem = os.path.splitext(filename)[0]

        if "_aug_" in stem:
            group = stem.split("_aug_")[0]
            groups[group] += 1
        else:
            unmatched.append(filename)
            groups[stem] += 1

    print("Total videos:", len(files))
    print("Likely original-video groups:", len(groups))
    print("Videos with _aug_ pattern:", len(files) - len(unmatched))
    print("Videos without _aug_ pattern:", len(unmatched))

    print("\nFirst 30 filenames:")

    for filename in files[:30]:
        print(filename)

    print("\nLargest 15 groups:")

    for group, count in groups.most_common(15):
        print(f"{group}: {count} videos")

    print("\nGroups containing only one video:",
          sum(1 for count in groups.values() if count == 1))

    if unmatched:
        print("\nExamples without augmentation pattern:")
        for filename in unmatched[:10]:
            print(filename)

print("\n" + "=" * 70)
print("AUGMENTATION ANALYSIS COMPLETED")
print("=" * 70)
