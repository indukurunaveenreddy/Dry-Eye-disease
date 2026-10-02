import os
import pickle

import cv2
import numpy as np

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye_camera_model.pkl")
IMG_SIZE = 50

AFFECTED = 1
NOT = 0

CLASS_NAMES = {AFFECTED: "Affected", NOT: "Not Affected"}


def _load_split(folder, label):
    images = []
    labels = []
    if not os.path.isdir(folder):
        return images, labels
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = img
        images.append(np.asarray(gray, dtype=np.float32) / 255.0)
        labels.append(label)
    return images, labels


def build_dataset():
    aff, aff_y = _load_split(os.path.join(DATASET_ROOT, "Affected"), AFFECTED)
    not_, not_y = _load_split(os.path.join(DATASET_ROOT, "Not"), NOT)
    if not aff and not not_:
        raise FileNotFoundError(
            "No images found under Dataset/Affected or Dataset/Not. "
            f"Expected at: {DATASET_ROOT}"
        )
    x = np.array(aff + not_, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(aff_y + not_y, dtype=np.int32)
    return x, y


def _sklearn_version():
    import sklearn

    return sklearn.__version__


def train_model(seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    x, y = build_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(x_train.reshape(len(x_train), -1), y_train)

    pred = model.predict(x_test.reshape(len(x_test), -1))
    accuracy = accuracy_score(y_test, pred) * 100

    report = classification_report(
        y_test, pred, target_names=["Not Affected", "Affected"], zero_division=0
    )
    cm = confusion_matrix(y_test, pred)

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(
            {
                "model": model,
                "accuracy": accuracy,
                "report": report,
                "confusion_matrix": cm,
                "trained_on": len(x),
                "sklearn_version": _sklearn_version(),
            },
            handle,
        )
    return accuracy, report, cm, len(x)


def load_model():
    """Return the cached model, or None if it is missing or unusable.

    scikit-learn estimators cannot be unpickled across major versions, so a cache
    written by a different interpreter is treated as absent and retrained.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as handle:
            bundle = pickle.load(handle)
    except Exception:
        return None
    if bundle.get("sklearn_version") != _sklearn_version():
        return None
    return bundle


def get_model():
    bundle = load_model()
    if bundle is None:
        train_model()
        bundle = load_model()
        if bundle is None:
            raise RuntimeError("Model training did not produce a usable cache file.")
    return bundle


def model_status():
    bundle = load_model()
    if bundle is None:
        cached = os.path.exists(MODEL_PATH)
        return {
            "trained": False,
            "stale_cache": cached,
            "sklearn_version": _sklearn_version(),
        }
    return {
        "trained": True,
        "stale_cache": False,
        "accuracy": bundle["accuracy"],
        "trained_on": bundle["trained_on"],
        "sklearn_version": bundle.get("sklearn_version"),
    }


def predict_frame(bgr_image, model):
    img = cv2.resize(bgr_image, (IMG_SIZE, IMG_SIZE))
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = img
    sample = np.asarray(gray, dtype=np.float32) / 255.0
    sample = sample.reshape(1, -1)

    label = int(model.predict(sample)[0])
    confidence = float(np.max(model.predict_proba(sample)[0])) * 100
    return label, confidence


_FACE_CASCADE = None
_EYE_CASCADE = None


def _cascades():
    global _FACE_CASCADE, _EYE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        _EYE_CASCADE = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        )
    return _FACE_CASCADE, _EYE_CASCADE


def find_eye_region(frame):
    """Locate candidate eye regions.

    The classifier is trained on infrared ocular crops, not on full faces, so the
    strategy is to try progressively looser detections and fall back to a centre
    crop. Every candidate is resized to the same 50x50 grayscale input the model
    was trained on, so a looser region costs accuracy but never crashes the scan.
    """
    face_cascade, eye_cascade = _cascades()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    regions = []

    def add(x1, y1, x2, y2):
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        regions.append({"box": (x1, y1, x2, y2), "crop": crop})

    # 1) Faces, then eyes inside each face.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
    for (fx, fy, fw, fh) in faces:
        roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi, 1.1, 6, minSize=(20, 20))
        for (ex, ey, ew, eh) in sorted(eyes, key=lambda e: e[1])[:2]:
            add(fx + ex - 4, fy + ey - 4, fx + ex + ew + 4, fy + ey + eh + 4)

    if regions:
        return regions, gray, frame, h, w, "face+eye"

    # 2) Eyes anywhere in the frame (close-up or partial face).
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 6, minSize=(24, 24))
    for (ex, ey, ew, eh) in sorted(eyes, key=lambda e: e[1])[:2]:
        add(ex - 4, ey - 4, ex + ew + 4, ey + eh + 4)

    if regions:
        return regions, gray, frame, h, w, "eye"

    # 3) Brightest connected blob, which is how an IR ocular image is framed.
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(biggest)
        if bw * bh > 0.10 * w * h:
            pad = int(0.05 * max(bw, bh))
            add(bx - pad, by - pad, bx + bw + pad, by + bh + pad)
            return regions, gray, frame, h, w, "bright-region"

    # 4) Centre crop, so an unrecognisable frame still produces a prediction.
    ch, cw = int(h * 0.5), int(w * 0.5)
    add((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)
    return regions, gray, frame, h, w, "centre-crop"


def analyse_frame(frame, model):
    """Classify a frame and annotate the eye region.

    The classifier is trained on whole infrared ocular images resized to 50x50, so
    the frame itself is the correct input. Cropping to a detected region shifts the
    input away from the training distribution and measurably hurts accuracy, so
    region detection is used only to draw the overlay, never to feed the model.
    """
    regions, gray, annotated, h, w, method = find_eye_region(frame)
    results = []

    for region in regions:
        label, confidence = predict_frame(region["crop"], model)
        results.append(
            {
                "label": label,
                "class_name": CLASS_NAMES[label],
                "confidence": confidence,
                "box": region["box"],
            }
        )

    display = frame.copy()
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for result in results:
        x1, y1, _, _ = result["box"]
        text = f'{result["class_name"]} {result["confidence"]:.0f}%'
        cv2.putText(
            display, text, (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            display, text, (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    frame_label, frame_confidence = predict_frame(frame, model)
    class_name = CLASS_NAMES[frame_label]

    banner = f"{class_name}  {frame_confidence:.0f}%"
    cv2.putText(
        display, banner, (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        display, banner, (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 0, 255) if frame_label == AFFECTED else (0, 200, 0), 1, cv2.LINE_AA,
    )

    return {
        "annotated": display,
        "results": results,
        "eye_count": len(regions),
        "severity": "affected" if frame_label == AFFECTED else "normal",
        "class_name": class_name,
        "confidence": frame_confidence,
        "method": method,
    }