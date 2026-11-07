
import os
import re
import cv2
import csv
import hashlib
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REAL_VIDEO_DIR = BASE_DIR / "SDFVD2.0_real"
FAKE_VIDEO_DIR = BASE_DIR / "SDFVD2.0_fake"

OUTPUT_DIR = BASE_DIR / "backend" / "experimental_sequences"

REAL_OUTPUT_DIR = OUTPUT_DIR / "real"
FAKE_OUTPUT_DIR = OUTPUT_DIR / "fake"

MANIFEST_PATH = OUTPUT_DIR / "experimental_manifest.csv"

SEQUENCE_LENGTH = 20
IMG_SIZE = 128

# First run: process only five videos per class.
# Change to None later to process the complete dataset.
SAMPLE_LIMIT = None

# Reject videos if fewer than this many sampled frames
# can be decoded successfully.
MIN_VALID_FRAMES = 15

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

REAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE HASHING
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# SOURCE GROUP IDENTIFICATION
# ============================================================

def get_source_group(filename, label):

    if label == 0:

        match = re.search(
            r"(?:real_)?v(\d+)",
            filename.lower()
        )

        if match:
            return "REAL_" + match.group(1)

    else:

        match = re.search(
            r"(?:fake_)?vs(\d+)",
            filename.lower()
        )

        if match:
            return "FAKE_" + match.group(1)

    return "UNKNOWN_" + Path(filename).stem


# ============================================================
# FRAME PREPROCESSING
# Matches backend inference preprocessing.
# ============================================================

def preprocess_frame(frame):

    # OpenCV BGR -> RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize to model input dimensions
    frame = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32
    frame = frame.astype(np.float32)

    # Normalize pixel values to [0, 1]
    frame /= 255.0

    return frame


# ============================================================
# VIDEO SEQUENCE EXTRACTION
# ============================================================

def extract_video_sequence(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise ValueError("Unable to open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        raise ValueError("Video contains no readable frames.")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    frames = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        try:

            processed_frame = preprocess_frame(frame)

            frames.append(processed_frame)

        except Exception as error:

            print(
                f"Frame preprocessing failed at index "
                f"{index}: {error}"
            )

    cap.release()

    valid_frames = len(frames)

    if valid_frames < MIN_VALID_FRAMES:

        raise ValueError(
            f"Only {valid_frames}/{SEQUENCE_LENGTH} "
            f"frames could be decoded. "
            f"Minimum required: {MIN_VALID_FRAMES}."
        )

    # Match inference behavior by repeating the last
    # successfully decoded frame when necessary.
    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            frames[-1].copy()
        )

    sequence = np.asarray(
        frames[:SEQUENCE_LENGTH],
        dtype=np.float32
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"Unexpected sequence shape: {sequence.shape}"
        )

    return sequence, valid_frames


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def get_video_files(folder):

    if not folder.exists():

        raise FileNotFoundError(
            f"Video folder not found: {folder}"
        )

    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files, key=lambda path: path.name.lower())


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(
    video_files,
    label,
    output_folder,
    class_name,
    seen_hashes,
    manifest_rows
):

    print("\n" + "=" * 65)
    print(f"PROCESSING {class_name.upper()} VIDEOS")
    print("=" * 65)

    processed = 0
    duplicates = 0
    failed = 0

    if SAMPLE_LIMIT is not None:

        grouped_videos = {}

        for video_path in video_files:

            group = get_source_group(
                video_path.name,
                label
            )

            if group not in grouped_videos:
                grouped_videos[group] = []

            grouped_videos[group].append(video_path)

        sorted_groups = sorted(
            grouped_videos.items(),
            key=lambda item: int(
                item[0].split("_")[-1]
            ) if item[0].split("_")[-1].isdigit()
            else 999999
        )

        video_files = [
            group_files[0]
            for group, group_files in sorted_groups[:SAMPLE_LIMIT]
        ]

        print(
            f"Distinct source groups selected: "
            f"{len(video_files)}"
        )

    print(f"Videos selected: {len(video_files)}")

    for video_number, video_path in enumerate(
        video_files,
        start=1
    ):

        print(
            f"\n[{video_number}/{len(video_files)}] "
            f"{video_path.name}"
        )

        file_hash = ""

        try:

            file_hash = calculate_sha256(video_path)

            # Exclude byte-for-byte duplicates within
            # the same class, without deleting source files.
            if file_hash in seen_hashes:

                duplicates += 1

                print("SKIPPED: Exact duplicate video.")

                manifest_rows.append({
                    "filename": video_path.name,
                    "class_name": class_name,
                    "label": label,
                    "source_group": get_source_group(
                        video_path.name,
                        label
                    ),
                    "sha256": file_hash,
                    "valid_frames": 0,
                    "sequence_path": "",
                    "status": "duplicate"
                })

                continue

            seen_hashes.add(file_hash)

            sequence, valid_frames = extract_video_sequence(
                video_path
            )

            # Use a hash prefix to avoid filename collisions.
            safe_name = re.sub(
                r"[^A-Za-z0-9_-]",
                "_",
                video_path.stem
            )

            output_name = (
                f"{class_name.lower()}_"
                f"{safe_name}_"
                f"{file_hash[:12]}.npy"
            )

            output_path = output_folder / output_name

            np.save(
                output_path,
                sequence
            )

            processed += 1

            print("SUCCESS")
            print(f"Shape: {sequence.shape}")
            print(f"Valid frames: {valid_frames}/20")
            print(
                f"Pixel range: "
                f"{sequence.min():.4f} to "
                f"{sequence.max():.4f}"
            )

            manifest_rows.append({
                "filename": video_path.name,
                "class_name": class_name,
                "label": label,
                "source_group": get_source_group(
                    video_path.name,
                    label
                ),
                "sha256": file_hash,
                "valid_frames": valid_frames,
                "sequence_path": str(
                    output_path.relative_to(BASE_DIR)
                ),
                "status": "processed"
            })

        except Exception as error:

            failed += 1

            print(f"FAILED: {error}")

            manifest_rows.append({
                "filename": video_path.name,
                "class_name": class_name,
                "label": label,
                "source_group": get_source_group(
                    video_path.name,
                    label
                ),
                "sha256": file_hash,
                "valid_frames": 0,
                "sequence_path": "",
                "status": "failed: " + str(error)
            })

    print("\nCLASS SUMMARY")
    print(f"Processed: {processed}")
    print(f"Duplicates skipped: {duplicates}")
    print(f"Failed: {failed}")

    return processed, duplicates, failed


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("EXPERIMENTAL DEEPFAKE PREPROCESSING")
    print("=" * 65)

    print(f"Project directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print("\nPreprocessing configuration:")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Image size: {IMG_SIZE} x {IMG_SIZE}")
    print("Colour format: RGB")
    print("Data type: float32")
    print("Normalization: [0, 1]")
    print(f"Minimum valid frames: {MIN_VALID_FRAMES}")
    print(f"Sample limit per class: {SAMPLE_LIMIT}")

    real_videos = get_video_files(REAL_VIDEO_DIR)
    fake_videos = get_video_files(FAKE_VIDEO_DIR)

    print("\nSOURCE DATASET")
    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")

    manifest_rows = []

    # Keep separate duplicate sets for the two classes.
    real_hashes = set()
    fake_hashes = set()

    real_summary = process_class(
        real_videos,
        label=0,
        output_folder=REAL_OUTPUT_DIR,
        class_name="REAL",
        seen_hashes=real_hashes,
        manifest_rows=manifest_rows
    )

    fake_summary = process_class(
        fake_videos,
        label=1,
        output_folder=FAKE_OUTPUT_DIR,
        class_name="FAKE",
        seen_hashes=fake_hashes,
        manifest_rows=manifest_rows
    )

    # Write processing manifest.
    fieldnames = [
        "filename",
        "class_name",
        "label",
        "source_group",
        "sha256",
        "valid_frames",
        "sequence_path",
        "status"
    ]

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 65)
    print("EXPERIMENTAL PREPROCESSING COMPLETE")
    print("=" * 65)

    print(f"Real summary: {real_summary}")
    print(f"Fake summary: {fake_summary}")

    print(
        f"\nManifest saved to:\n{MANIFEST_PATH}"
    )

    print(
        f"\nReal sequences saved to:\n{REAL_OUTPUT_DIR}"
    )

    print(
        f"\nFake sequences saved to:\n{FAKE_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
