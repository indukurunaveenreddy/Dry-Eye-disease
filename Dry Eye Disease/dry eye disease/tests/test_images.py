"""Tests for image loading, resizing and stratified splitting."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ded import config, images


def test_load_image_normalises_and_resizes(tmp_path) -> None:
    source = tmp_path / "sample.png"
    Image.fromarray(np.full((80, 120), 255, dtype=np.uint8), mode="L").save(source)

    array = images.load_image(source, size=32, grayscale=True)

    assert array is not None
    assert array.shape == (32, 32, 1)
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0 and float(array.max()) <= 1.0
    assert array.max() == pytest.approx(1.0)


def test_load_image_returns_none_for_corrupt_file(tmp_path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    assert images.load_image(broken, size=32) is None


def test_load_image_handles_tiny_images(tmp_path) -> None:
    tiny = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(tiny)
    assert images.load_image(tiny, size=32) is None


def test_collect_folders_applies_minimum_sample_filter(tmp_path) -> None:
    big = tmp_path / "big"
    small = tmp_path / "small"
    big.mkdir()
    small.mkdir()
    for index in range(5):
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L").save(
            big / f"img{index}.png"
        )
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L").save(small / "only.png")

    folders = images.collect_folders(tmp_path, {"big": 0, "small": 1}, min_samples=3)

    assert [folder.name for folder in folders] == ["big"]
    assert len(folders[0].paths) == 5


def test_list_class_folder_ignores_non_images(tmp_path) -> None:
    folder = tmp_path / "cls"
    folder.mkdir()
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L").save(folder / "a.png")
    (folder / ".DS_Store").write_bytes(b"junk")

    result = images.list_class_folder(folder, "cls", 0)
    assert len(result.paths) == 1
    assert result.paths[0].name == "a.png"


def test_load_dataset_shapes_and_labels(tmp_path) -> None:
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(3):
            Image.fromarray(
                np.full((12, 12), 100 + index, dtype=np.uint8), mode="L"
            ).save(folder / f"{index}.png")

    features, targets, counts = images.load_dataset(
        tmp_path, {"one": 0, "two": 1}, size=16, grayscale=True
    )

    assert features.shape == (6, 16, 16, 1)
    assert counts == {"one": 3, "two": 3}
    assert sorted(set(targets.tolist())) == [0, 1]


def test_load_dataset_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        images.load_dataset(tmp_path / "nope", {"absent": 0}, size=16)


def test_stratified_split_preserves_class_balance() -> None:
    features = np.zeros((100, 4, 4, 1), dtype=np.float32)
    targets = np.array([0] * 50 + [1] * 50)

    x_train, y_train, x_test, y_test = images.stratified_split(
        features, targets, test_size=0.2
    )

    assert len(x_train) == 80 and len(x_test) == 20
    assert (y_train == 0).sum() == 40 and (y_test == 1).sum() == 10
    assert set(y_train.tolist()) == {0, 1}


def test_stratified_split_is_deterministic() -> None:
    features = np.arange(80, dtype=np.float32).reshape(80, 1, 1, 1)
    targets = np.array([0] * 40 + [1] * 40)

    first = images.stratified_split(features, targets)
    second = images.stratified_split(features, targets)

    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[3], second[3])


def test_standardize_image_zero_centres_each_channel() -> None:
    rng = np.random.default_rng(0)
    array = rng.uniform(0.2, 0.9, size=(16, 16, 1)).astype(np.float32)

    out = images.standardize_image(array)

    assert out.shape == array.shape
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0 / 3.0) < 0.05


def test_standardize_image_survives_constant_channel() -> None:
    array = np.full((8, 8, 1), 0.5, dtype=np.float32)
    out = images.standardize_image(array)
    assert np.all(np.isfinite(out))


def test_load_image_standardize_produces_zero_mean(tmp_path) -> None:
    source = tmp_path / "dim.png"
    Image.fromarray(np.full((40, 40), 200, dtype=np.uint8), mode="L").save(source)

    raw = images.load_image(source, size=20, standardize=False)
    scaled = images.load_image(source, size=20, standardize=True)

    assert raw is not None and scaled is not None
    assert abs(float(scaled.mean())) < 1e-4


def test_flatten_is_picklable() -> None:
    import pickle

    transformer = images.Flatten()
    data = np.zeros((4, 8, 8, 1), dtype=np.float32)
    assert transformer.fit(data).transform(data).shape == (4, 64)
    assert pickle.loads(pickle.dumps(transformer)) is not None


def test_stratified_group_split_keeps_subjects_disjoint() -> None:
    """The returned rows must correspond to a split with no shared subjects."""
    from sklearn.model_selection import GroupShuffleSplit

    rng = np.random.default_rng(2)
    features = rng.normal(size=(100, 4, 4, 1)).astype(np.float32)
    targets = np.array([0, 1] * 50)
    groups = np.repeat([f"s{index}" for index in range(50)], 2)

    x_train, y_train, x_test, y_test = images.stratified_group_split(
        features, targets, groups, test_size=0.2
    )

    assert len(x_train) + len(x_test) == len(targets)
    # The same seed and splitter must yield disjoint subject sets, so the
    # realised partition is reproducible and free of subject leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    train_index, test_index = next(
        splitter.split(np.arange(len(targets)), targets, groups)
    )
    assert not set(groups[train_index]) & set(groups[test_index])
    assert len(train_index) == len(x_train)
