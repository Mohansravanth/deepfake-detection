import cv2
import numpy as np
import os
import gc


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

IMG_SIZE = 128

MAX_FRAMES = 30

FRAME_SKIP = 6

MAX_DIMENSION = 320


# ============================================================
# FACE EXTRACTION
# ============================================================

def extract_faces_haar(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = []

    frame_count = 0

    while (
        len(faces) < SEQUENCE_LENGTH
        and frame_count < MAX_FRAMES
    ):

        ret, frame = cap.read()

        if not ret:

            break

        frame_count += 1

        # Skip frames
        if frame_count % FRAME_SKIP != 0:

            continue

        # ============================================
        # RESIZE LARGE FRAME
        # ============================================

        try:

            h, w = frame.shape[:2]

            if max(h, w) > MAX_DIMENSION:

                scale = (
                    MAX_DIMENSION /
                    max(h, w)
                )

                new_w = int(
                    w * scale
                )

                new_h = int(
                    h * scale
                )

                frame = cv2.resize(
                    frame,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

        except Exception:

            continue

        # ============================================
        # FACE DETECTION
        # ============================================

        try:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

        except Exception:

            continue

        if len(detected) == 0:

            continue

        # ============================================
        # LARGEST FACE
        # ============================================

        x, y, w, h = max(
            detected,
            key=lambda rect:
                rect[2] * rect[3]
        )

        face = frame[
            y:y+h,
            x:x+w
        ]

        if face.size == 0:

            continue

        try:

            face = cv2.resize(
                face,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            faces.append(
                face
            )

        except Exception:

            continue

    cap.release()

    gc.collect()

    # ================================================
    # PAD IF FEWER THAN 20 FACES
    # ================================================

    if len(faces) > 0:

        while len(faces) < SEQUENCE_LENGTH:

            faces.append(
                faces[-1].copy()
            )

    if len(faces) == 0:

        return None

    return np.asarray(
        faces,
        dtype=np.uint8
    )


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REAL_DIR = os.path.join(
    BASE_DIR,
    "SDFVD2.0_real"
)

FAKE_DIR = os.path.join(
    BASE_DIR,
    "SDFVD2.0_fake"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "processed_sequences"
)

REAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "real"
)

FAKE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "fake"
)

os.makedirs(
    REAL_OUTPUT,
    exist_ok=True
)

os.makedirs(
    FAKE_OUTPUT,
    exist_ok=True
)


# ============================================================
# PROCESS FUNCTION
# ============================================================

def process_directory(
    input_dir,
    output_dir,
    label_name
):

    if not os.path.exists(input_dir):

        print(
            f"ERROR: Directory not found:"
            f"\n{input_dir}"
        )

        return 0, 0

    saved = 0
    skipped = 0

    videos = sorted(
        os.listdir(input_dir)
    )

    for index, video in enumerate(videos):

        if not video.lower().endswith(
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm"
            )
        ):

            continue

        video_path = os.path.join(
            input_dir,
            video
        )

        print(
            f"[{index + 1}] "
            f"Processing {label_name}: "
            f"{video}"
        )

        try:

            sequence = extract_faces_haar(
                video_path
            )

            if (
                sequence is None
                or len(sequence) != SEQUENCE_LENGTH
            ):

                print(
                    "   SKIPPED - insufficient faces"
                )

                skipped += 1

                continue

            # Use original filename
            base_name = os.path.splitext(
                video
            )[0]

            output_path = os.path.join(
                output_dir,
                base_name + ".npy"
            )

            np.save(
                output_path,
                sequence
            )

            print(
                f"   SAVED -> {output_path}"
            )

            saved += 1

        except Exception as e:

            print(
                f"   ERROR: {e}"
            )

            skipped += 1

    return saved, skipped


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DEEPFAKE DATASET PREPROCESSOR")
    print("=" * 60)

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Image size      : {IMG_SIZE}x{IMG_SIZE}"
    )

    print(
        f"Maximum frames  : {MAX_FRAMES}"
    )

    print(
        f"Frame skip      : {FRAME_SKIP}"
    )

    print("=" * 60)

    print("\nProcessing REAL videos...")

    real_saved, real_skipped = process_directory(
        REAL_DIR,
        REAL_OUTPUT,
        "REAL"
    )

    print("\nProcessing FAKE videos...")

    fake_saved, fake_skipped = process_directory(
        FAKE_DIR,
        FAKE_OUTPUT,
        "FAKE"
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"REAL saved   : {real_saved}"
    )

    print(
        f"REAL skipped : {real_skipped}"
    )

    print(
        f"FAKE saved   : {fake_saved}"
    )

    print(
        f"FAKE skipped : {fake_skipped}"
    )

    print(
        f"TOTAL        : "
        f"{real_saved + fake_saved}"
    )

    print("=" * 60)