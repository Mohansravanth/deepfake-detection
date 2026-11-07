import os
import re

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REAL_DIR = os.path.join(BASE_DIR, "SDFVD2.0_real")
FAKE_DIR = os.path.join(BASE_DIR, "SDFVD2.0_fake")

EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def extract_group(filename, category):

    stem = os.path.splitext(filename)[0]

    if category == "REAL":
        match = re.search(r"real_v(\d+)", stem, re.IGNORECASE)
        if not match:
            match = re.fullmatch(r"v(\d+)", stem, re.IGNORECASE)

    else:
        match = re.search(r"fake_vs(\d+)", stem, re.IGNORECASE)
        if not match:
            match = re.fullmatch(r"vs(\d+)", stem, re.IGNORECASE)

    return int(match.group(1)) if match else None


def collect_groups(folder, category):

    groups = {}

    for filename in os.listdir(folder):

        if not filename.lower().endswith(EXTENSIONS):
            continue

        group_id = extract_group(filename, category)

        if group_id is not None:
            groups.setdefault(group_id, []).append(filename)

    return groups


real_groups = collect_groups(REAL_DIR, "REAL")
fake_groups = collect_groups(FAKE_DIR, "FAKE")

real_ids = set(real_groups)
fake_ids = set(fake_groups)

common = sorted(real_ids & fake_ids)
real_only = sorted(real_ids - fake_ids)
fake_only = sorted(fake_ids - real_ids)

print("=" * 65)
print("REAL AND FAKE GROUP MATCHING")
print("=" * 65)

print("Real group IDs:", len(real_ids))
print("Fake group IDs:", len(fake_ids))
print("Matching IDs:", len(common))
print("Real-only IDs:", len(real_only))
print("Fake-only IDs:", len(fake_only))

print("\nFirst 20 matching groups:")

for group_id in common[:20]:

    print(f"\nGROUP {group_id}")

    print("REAL:")
    for filename in real_groups[group_id][:5]:
        print(" ", filename)

    print("FAKE:")
    for filename in fake_groups[group_id][:5]:
        print(" ", filename)

print("\nReal-only IDs:", real_only[:30])
print("Fake-only IDs:", fake_only[:30])

print("\n" + "=" * 65)
print("GROUP MATCHING COMPLETED")
print("=" * 65)
