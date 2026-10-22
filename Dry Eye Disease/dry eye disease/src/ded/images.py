"""Image loading utilities shared by the CNN models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif", ".gif", ".tiff"}
MAX_WORKERS = 8


@dataclass(frozen=True)
class ImageFolder:
    """One class folder of images on disk."""

    name: str
    label: int
    paths: tuple[Path, ...]


def list_class_folder(directory: Path, name: str, label: int) -> ImageFolder:
    """Collect readable image paths for a single class directory."""
    if not directory.is_dir():
        return ImageFolder(name=name, label=label, paths=())

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return ImageFolder(name=name, label=label, paths=tuple(paths))


def collect_folders(
    root: Path,
    class_map: dict[str, int],
    min_samples: int = 0,
) -> list[ImageFolder]:
    """Enumerate class folders, skipping ones that are too small to split."""
    folders = [
        list_class_folder(root / name, name, label)
        for name, label in class_map.items()
        if (root / name).is_dir()
    ]
    return [folder for folder in folders if len(folder.paths) >= min_samples]


def load_image(
    path: Path,
    size: int,
    grayscale: bool = True,
    standardize: bool = False,
) -> np.ndarray | None:
    """Read one image as a normalised float32 array, or None if unreadable.

    With ``standardize`` the image is z-scored per channel, which matters for
    the infrared eye photographs whose global brightness varies widely
    between subjects.
    """
    try:
        with Image.open(path) as handle:
            handle.load()
            if grayscale:
                image = handle.convert("L")
                array = np.asarray(image, dtype=np.float32)
                array = array[..., None]
            else:
                image = handle.convert("RGB")
                array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    height, width = array.shape[:2]
    if height < 4 or width < 4:
        return None

    resized = np.asarray(
        Image.fromarray(array.squeeze() if grayscale else array.astype(np.uint8)).resize(
            (size, size), Image.BILINEAR
        ),
        dtype=np.float32,
    )
    if grayscale:
        resized = resized[..., None]

    array_out = resized / 255.0
    if standardize:
        array_out = standardize_image(array_out)
    return array_out


def standardize_image(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each channel of a single image, mapping to roughly [-1, 1].

    Accumulated in float64: a nearly uniform infrared frame has a standard
    deviation near zero, and doing this in float32 divides rounding noise by
    that tiny value and produces a visibly wrong result.
    """
    data = np.asarray(array, dtype=np.float64)
    mean = data.mean(axis=(0, 1), keepdims=True)
    std = data.std(axis=(0, 1), keepdims=True)
    scale = np.where(std < eps, 1.0, std)
    scaled = np.clip((data - mean) / scale, -3.0, 3.0) / 3.0
    return scaled.astype(np.float32)


def load_folder_batch(
    folder: ImageFolder,
    size: int,
    grayscale: bool = True,
    workers: int = MAX_WORKERS,
    standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Load an entire class folder into memory, dropping unreadable files."""
    def read(path: Path) -> np.ndarray | None:
        return load_image(path, size, grayscale, standardize)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(pool.map(read, folder.paths))

    kept_images: list[np.ndarray] = []
    kept_paths: list[Path] = []
    for path, array in zip(folder.paths, arrays, strict=True):
        if array is not None:
            kept_images.append(array)
            kept_paths.append(path)

    if not kept_images:
        return (
            np.zeros((0, size, size, 1 if grayscale else 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            [],
        )

    images = np.stack(kept_images).astype(np.float32)
    labels = np.full(len(images), folder.label, dtype=np.int32)
    return images, labels, kept_paths


def load_dataset(
    root: Path,
    class_map: dict[str, int],
    size: int,
    grayscale: bool = True,
    min_samples: int = 0,
    standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load every class folder under ``root`` into one stacked array."""
    folders = collect_folders(root, class_map, min_samples=min_samples)
    if not folders:
        raise FileNotFoundError(
            f"No usable class folders under {root}. Expected: {sorted(class_map)}"
        )

    all_images: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    counts: dict[str, int] = {}

    for folder in folders:
        images, labels, _ = load_folder_batch(folder, size, grayscale, standardize=standardize)
        all_images.append(images)
        all_labels.append(labels)
        counts[folder.name] = len(images)

    features = np.concatenate(all_images, axis=0)
    targets = np.concatenate(all_labels, axis=0)
    return features, targets, counts


def stratified_split(
    features: np.ndarray,
    targets: np.ndarray,
    test_size: float = 0.2,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified train/test split."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(targets))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=targets,
    )
    return (
        features[train_index],
        targets[train_index],
        features[test_index],
        targets[test_index],
    )


class Flatten(BaseEstimator, TransformerMixin):
    """Flatten image tensors into feature vectors for linear models.

    Defined at module level so fitted pipelines containing it can be pickled.
    """

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Flatten:
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x).reshape(len(x), -1)


def stratified_group_split(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    seed: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by subject so no subject appears in both train and test.

    Several images per eye are captured per subject, so a plain stratified
    split leaks subject identity across the boundary.
    """
    from sklearn.model_selection import GroupShuffleSplit

    indices = np.arange(len(targets))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(indices, targets, groups))
    return (
        features[train_index],
        targets[train_index],
        features[test_index],
        targets[test_index],
    )
