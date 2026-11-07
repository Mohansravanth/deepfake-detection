import cv2
import numpy as np

SEQUENCE_LENGTH = 20
IMG_SIZE = 128
FEATURE_NAMES = [
    "temporal_mean",
    "temporal_std",
    "temporal_max",
    "sharpness_mean",
    "sharpness_std",
    "fft_high_ratio",
    "fft_mid_ratio",
    "color_r_mean",
    "color_g_mean",
    "color_b_mean",
    "color_r_std",
    "color_g_std",
    "color_b_std",
    "center_border_luma_diff",
    "center_border_std_diff",
    "noise_std",
    "chroma_corr_rg",
    "chroma_corr_rb",
    "block_energy",
    "face_coverage",
    "mouth_temporal",
    "eye_temporal",
    "saturation_mean",
    "saturation_std",
]


def _to_unit_interval(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.max() > 1.5:
        sequence = sequence / 255.0
    return np.clip(sequence, 0.0, 1.0)


def extract_forensic_vector(sequence, face_coverage=0.0):
    sequence = _to_unit_interval(sequence)
    if sequence.ndim != 4:
        raise ValueError("Expected sequence shape (T, H, W, C).")

    gray = sequence.mean(axis=-1)
    if gray.shape[0] > 1:
        diffs = np.abs(np.diff(gray, axis=0))
        temporal_mean = float(diffs.mean())
        temporal_std = float(diffs.std())
        temporal_max = float(diffs.max())
    else:
        temporal_mean = temporal_std = temporal_max = 0.0

    lap_vars = []
    for frame in gray:
        laplacian = cv2.Laplacian((frame * 255).astype(np.uint8), cv2.CV_64F)
        lap_vars.append(float(laplacian.var()))
    sharpness_mean = float(np.mean(lap_vars)) if lap_vars else 0.0
    sharpness_std = float(np.std(lap_vars)) if lap_vars else 0.0

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray.mean(axis=0))))
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    radius = np.sqrt(
        (np.arange(height)[:, None] - center_y) ** 2
        + (np.arange(width)[None, :] - center_x) ** 2
    )
    total = float(spectrum.sum() + 1e-8)
    fft_high_ratio = float(spectrum[radius > 24].sum() / total)
    fft_mid_ratio = float(spectrum[(radius > 8) & (radius <= 24)].sum() / total)

    color_means = sequence.mean(axis=(0, 1, 2))
    color_stds = sequence.std(axis=(0, 1, 2))

    h, w = gray.shape[1:]
    center = gray[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    border = np.concatenate(
        [
            gray[:, : h // 8, :].reshape(gray.shape[0], -1),
            gray[:, -h // 8 :, :].reshape(gray.shape[0], -1),
            gray[:, :, : w // 8].reshape(gray.shape[0], -1),
            gray[:, :, -w // 8 :].reshape(gray.shape[0], -1),
        ],
        axis=1,
    )
    center_border_luma_diff = float(center.mean() - border.mean())
    center_border_std_diff = float(center.std() - border.std())

    blurred = np.stack(
        [cv2.GaussianBlur((frame * 255).astype(np.uint8), (5, 5), 0) for frame in gray]
    ).astype(np.float32) / 255.0
    noise_std = float(np.std(gray - blurred))

    flat = sequence.reshape(-1, 3)
    if len(flat) > 8:
        chroma_corr_rg = float(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1])
        chroma_corr_rb = float(np.corrcoef(flat[:, 0], flat[:, 2])[0, 1])
    else:
        chroma_corr_rg = chroma_corr_rb = 0.0

    block = gray[:, ::8, :][:, :, ::8]
    neighbors = gray[:, 4::8, :][:, :, 4::8]
    min_t = min(block.shape[0], neighbors.shape[0])
    min_h = min(block.shape[1], neighbors.shape[1])
    min_w = min(block.shape[2], neighbors.shape[2])
    block_energy = float(
        np.mean(np.abs(block[:min_t, :min_h, :min_w] - neighbors[:min_t, :min_h, :min_w]))
    )

    mouth = gray[:, int(h * 0.62) : int(h * 0.92), int(w * 0.25) : int(w * 0.75)]
    eyes = gray[:, int(h * 0.18) : int(h * 0.48), int(w * 0.12) : int(w * 0.88)]
    mouth_temporal = float(np.abs(np.diff(mouth, axis=0)).mean()) if mouth.shape[0] > 1 else 0.0
    eye_temporal = float(np.abs(np.diff(eyes, axis=0)).mean()) if eyes.shape[0] > 1 else 0.0

    hsv = np.stack(
        [
            cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            for frame in sequence
        ]
    )
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    values = np.array(
        [
            temporal_mean,
            temporal_std,
            temporal_max,
            sharpness_mean,
            sharpness_std,
            fft_high_ratio,
            fft_mid_ratio,
            float(color_means[0]),
            float(color_means[1]),
            float(color_means[2]),
            float(color_stds[0]),
            float(color_stds[1]),
            float(color_stds[2]),
            center_border_luma_diff,
            center_border_std_diff,
            noise_std,
            chroma_corr_rg if np.isfinite(chroma_corr_rg) else 0.0,
            chroma_corr_rb if np.isfinite(chroma_corr_rb) else 0.0,
            block_energy,
            float(face_coverage),
            mouth_temporal,
            eye_temporal,
            float(saturation.mean()),
            float(saturation.std()),
        ],
        dtype=np.float32,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def detect_face_crop(frame_bgr, target_size=IMG_SIZE):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0, 0.0

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    pad = int(0.25 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)
    crop = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    coverage = float((w * h) / max(frame_bgr.shape[0] * frame_bgr.shape[1], 1))
    return resized.astype(np.float32) / 255.0, coverage


def extract_media_sequence(path, sequence_length=SEQUENCE_LENGTH):
    image = cv2.imread(path)
    if image is not None and path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        frame, coverage = detect_face_crop(image)
        sequence = np.repeat(frame[np.newaxis, ...], sequence_length, axis=0)
        return sequence.astype(np.float32), float(coverage)

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("Unable to open media file.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError("Video has no readable frames.")

    indices = np.linspace(0, total - 1, sequence_length).astype(int)
    frames = []
    coverages = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        crop, coverage = detect_face_crop(frame)
        frames.append(crop)
        coverages.append(coverage)
    capture.release()

    if not frames:
        raise ValueError("Could not extract frames.")

    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())
        coverages.append(coverages[-1] if coverages else 0.0)

    sequence = np.asarray(frames[:sequence_length], dtype=np.float32)
    return sequence, float(np.mean(coverages[:sequence_length]))
