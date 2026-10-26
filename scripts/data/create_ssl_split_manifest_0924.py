"""Recreate the approved 6000/6008/3014/3005 nuScenes SSL split manifest."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))


import argparse
import hashlib
import json
import sys
from pathlib import Path

from nuscenes.utils.splits import create_splits_scenes


def read_tokens(path):
    values = path.read_text(encoding="utf-8").split()
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate tokens in {path}")
    return set(values)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(str(REPO_ROOT)))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    from scripts.data.generate_ssl_pkls import (  # noqa: E402
        build_val_log_stats,
        choose_balanced_val_logs,
        metadata_indices,
    )

    data_root = args.data_root
    selection = data_root / "scene_selection_6000_0916"
    comparison = json.loads((selection / "selection_comparison.json").read_text())
    selected_name = comparison["recommended_candidate"]
    if selected_name != "balanced_150scenes_6000samples":
        raise RuntimeError(f"Unexpected selected candidate: {selected_name}")
    scene_path = selection / f"{selected_name}_scene_tokens.txt"
    sample_path = selection / f"{selected_name}_sample_tokens.txt"
    labeled_scenes = read_tokens(scene_path)
    labeled_tokens = read_tokens(sample_path)
    if len(labeled_scenes) != 150 or len(labeled_tokens) != 6000:
        raise RuntimeError("Labeled selection is not 150 complete scenes / 6000 samples")

    version_root, logs, scenes, samples, _ = metadata_indices(data_root)
    train_names = set(create_splits_scenes()["train"])
    val_names = set(create_splits_scenes()["val"])
    if any(scenes[token]["name"] not in train_names for token in labeled_scenes):
        raise RuntimeError("Labeled selection contains a non-train scene")
    expected_labeled = {s["token"] for s in samples if s["scene_token"] in labeled_scenes}
    if expected_labeled != labeled_tokens:
        raise RuntimeError("Labeled sample list does not contain exactly whole scenes")

    stats = build_val_log_stats(version_root, logs, scenes, samples)
    val_logs, internal_test_logs = choose_balanced_val_logs(stats)
    val_log_set = set(val_logs)
    internal_test_log_set = set(internal_test_logs)
    val_scenes = {token for token, row in scenes.items() if row["name"] in val_names}
    validation_scenes = {token for token in val_scenes if scenes[token]["log_token"] in val_log_set}
    internal_test_scenes = {token for token in val_scenes if scenes[token]["log_token"] in internal_test_log_set}
    validation_tokens = {s["token"] for s in samples if s["scene_token"] in validation_scenes}
    internal_test_tokens = {s["token"] for s in samples if s["scene_token"] in internal_test_scenes}
    if len(validation_tokens) != 3014 or len(internal_test_tokens) != 3005:
        raise RuntimeError(
            f"Unexpected val/internal-test counts: {len(validation_tokens)}/{len(internal_test_tokens)}"
        )
    if validation_scenes & internal_test_scenes or validation_tokens & internal_test_tokens:
        raise RuntimeError("Validation and internal-test splits overlap")
    if len(validation_scenes | internal_test_scenes) != 150:
        raise RuntimeError("Official validation scenes are not fully partitioned")

    manifest = {
        "description": "MV2DFusion SSL: 150 whole labeled train scenes, official test unlabeled, official val split by whole log",
        "labeled_train": {
            "scene_tokens": sorted(labeled_scenes),
            "sample_tokens": sorted(labeled_tokens),
            "selection_sha256": file_hash(scene_path),
        },
        "unlabeled_train": {
            "source": "official nuScenes v1.0-test",
            "expected_scenes": 150,
            "expected_samples": 6008,
        },
        "validation": {
            "log_tokens": sorted(val_log_set),
            "scene_tokens": sorted(validation_scenes),
            "sample_tokens": sorted(validation_tokens),
        },
        "internal_test": {
            "log_tokens": sorted(internal_test_log_set),
            "scene_tokens": sorted(internal_test_scenes),
            "sample_tokens": sorted(internal_test_tokens),
        },
    }
    output = data_root / "ssl_splits" / "nuscenes_ssl_split_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"MANIFEST_OK labeled={len(labeled_tokens)} "
        f"validation={len(validation_tokens)} internal_test={len(internal_test_tokens)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
