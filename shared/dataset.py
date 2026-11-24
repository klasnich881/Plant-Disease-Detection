import json
import random
from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
]


# Loads the raw PlantVillage tomato dataset across the 5 target classes,
# randomly samples 1,000 images per class using a fixed seed for class balance,
# splits each class 70/15/15 into train/val/test sets, and saves the split manifest to disk.
def create_and_save_splits(
    raw_data_dir: str | Path,
    split_output_path: str | Path,
    class_names: list[str] = DEFAULT_CLASS_NAMES,
    samples_per_class: int = 1000,
    random_seed: int = 42,
) -> dict:
    raw_dir = Path(raw_data_dir)
    output_path = Path(split_output_path)
    project_root = output_path.resolve().parent.parent.parent

    rng = random.Random(random_seed)

    splits = {
        "random_seed": random_seed,
        "samples_per_class": samples_per_class,
        "class_names": class_names,
        "train": {"paths": [], "labels": []},
        "val": {"paths": [], "labels": []},
        "test": {"paths": [], "labels": []},
    }

    n_train = int(samples_per_class * 0.70)  # 700
    n_val = int(samples_per_class * 0.15)    # 150
    # Remaining 15% goes to test (150)

    for label_idx, class_name in enumerate(class_names):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        # Sort first so directory traversal order is consistent
        all_images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        if len(all_images) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' only has {len(all_images)} images, "
                f"but {samples_per_class} are required."
            )

        sampled_images = rng.sample(all_images, samples_per_class)

        train_imgs = sampled_images[:n_train]
        val_imgs = sampled_images[n_train : n_train + n_val]
        test_imgs = sampled_images[n_train + n_val :]

        for split_name, img_list in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            for img_path in img_list:
                # Store paths relative to project root for cross-computer portability
                rel_path = img_path.resolve().relative_to(project_root).as_posix()
                splits[split_name]["paths"].append(rel_path)
                splits[split_name]["labels"].append(label_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return splits


# Loads the pre-saved 70/15/15 dataset split manifest from disk so Approach A
# and Approach B operate on the exact same training, validation, and testing images.
def load_splits(
    split_path: str | Path,
    project_root: str | Path | None = None,
) -> dict:
    split_file = Path(split_path).resolve()
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {split_file}. Run create_and_save_splits() first."
        )

    root = Path(project_root).resolve() if project_root else split_file.parent.parent.parent

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Resolve relative paths back to valid local filesystem paths
    for split_name in ("train", "val", "test"):
        splits[split_name]["paths"] = [
            str(root / rel_path) for rel_path in splits[split_name]["paths"]
        ]

    return splits