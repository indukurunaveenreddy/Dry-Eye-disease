"""Build a class-capped, size-reduced copy of the image datasets.

Training on every image is wasteful here: the blink corpus holds about 3,500
images per gaze class against 109 for the eye-state classes, which slows each
epoch and lets the network memorise near-duplicate crops. Each class is capped
at ``MAX_IMAGES_PER_CLASS``.

Images are also rewritten at the size each model actually consumes, as
grayscale JPEG, rather than symlinked. The source corpus is 235 MB of infrared
frames at up to 1280x640; the models only ever see them at 96x96 and 64x64, so
resizing on the way in removes roughly 99 percent of the footprint without
changing a single input tensor.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ded import config, images  # noqa: E402


def compress_file(source: Path, destination: Path, size: int, quality: int) -> None:
    """Rewrite one image as a grayscale JPEG at the model's input size."""
    with Image.open(source) as handle:
        handle.load()
        image = handle.convert("L").resize((size, size), Image.BILINEAR)
        image.save(destination, "JPEG", quality=quality, optimize=True)


def build_subset(
    source_root: Path,
    destination_root: Path,
    class_map: dict[str, int],
    cap: int,
    size: int,
    quality: int = 85,
    seed: int = config.RANDOM_STATE,
    compress: bool = True,
) -> dict[str, int]:
    """Write up to ``cap`` images per class into ``destination_root``."""
    rng = random.Random(seed)
    destination_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    written_bytes = 0

    for name in class_map:
        source_dir = source_root / name
        if not source_dir.is_dir():
            print(f"  skipping {name}: {source_dir} is missing")
            continue

        available = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in images.IMAGE_SUFFIXES
        )
        chosen = available
        if len(available) > cap:
            chosen = sorted(rng.sample(available, cap))

        target_dir = destination_root / name
        if target_dir.exists():
            for stale in target_dir.iterdir():
                stale.unlink()
        else:
            target_dir.mkdir(parents=True)

        kept = 0
        for path in chosen:
            target = target_dir / f"{path.stem}.jpg"
            try:
                if compress:
                    compress_file(path, target, size, quality)
                else:
                    target.symlink_to(path.resolve())
            except (OSError, ValueError):
                continue
            written_bytes += target.stat().st_size
            kept += 1

        counts[name] = kept
        note = f" (of {len(available)})" if kept != len(available) else ""
        print(f"  {name:14s} {kept:>4d}{note}")

    if compress and written_bytes:
        print(f"  -> {written_bytes / 1024 / 1024:.1f} MB written for this corpus")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=config.MAX_IMAGES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument(
        "--quality", type=int, default=85, help="JPEG quality for compressed output."
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of resizing and compressing.",
    )
    args = parser.parse_args()

    compress = not args.symlink
    mode = "symlink" if args.symlink else f"resize + JPEG q{args.quality}"
    print(f"Building subset: at most {args.cap} images per class, {mode}\n")

    print(f"Eye disease corpus ({config.EYE_IMAGE_SIZE}px)")
    eye_counts = build_subset(
        config.EYE_IMAGE_ROOT,
        config.BALANCED_EYE_ROOT,
        config.EYE_CLASSES,
        args.cap,
        config.EYE_IMAGE_SIZE,
        args.quality,
        args.seed,
        compress,
    )

    print(f"\nBlink and gaze corpus ({config.BLINK_IMAGE_SIZE}px)")
    blink_counts = build_subset(
        config.BLINK_IMAGE_ROOT,
        config.BALANCED_BLINK_ROOT,
        config.BLINK_CLASSES,
        args.cap,
        config.BLINK_IMAGE_SIZE,
        args.quality,
        args.seed,
        compress,
    )

    eye_total = sum(eye_counts.values())
    blink_total = sum(blink_counts.values())

    inventory = config.DATA_PROCESSED / "balanced_inventory.csv"
    lines = ["task,class,images"]
    for name, value in eye_counts.items():
        lines.append(f"eye disease,{name},{value}")
    for name, value in blink_counts.items():
        lines.append(f"blink / gaze,{name},{value}")
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n  eye:   {eye_total} images across {len(eye_counts)} classes")
    print(f"  blink: {blink_total} images across {len(blink_counts)} classes")
    print(f"  total: {eye_total + blink_total}")
    print(f"  wrote {inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
