import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "REAL": os.path.join(BASE_DIR, "SDFVD2.0_real"),
    "FAKE": os.path.join(BASE_DIR, "SDFVD2.0_fake")
}

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

for category, folder in DATASETS.items():

    print("\n" + "=" * 65)
    print(f"{category} VIDEO DIAGNOSTICS")
    print("=" * 65)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    videos = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )

    print("Total videos:", len(videos))

    for filename in videos[:5]:

        path = os.path.join(folder, filename)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print("\n", filename, "- Cannot open video")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        sample_indices = np.linspace(
            0,
            max(frame_count - 1, 0),
            min(20, frame_count),
            dtype=int
        )

        detected_faces = 0
        readable_frames = 0

        for frame_index in sample_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()

            if not ret:
                continue

            readable_frames += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                detected_faces += 1

        cap.release()

        print("\nVideo:", filename)
        print("Resolution:", width, "x", height)
        print("Total frames:", frame_count)
        print("FPS:", round(fps, 2))
        print("Duration:", round(duration, 2), "seconds")
        print("Readable sampled frames:", readable_frames)
        print("Frames with detected faces:", detected_faces, "/", readable_frames)

print("\n" + "=" * 65)
print("DIAGNOSTICS COMPLETED")
print("=" * 65)

