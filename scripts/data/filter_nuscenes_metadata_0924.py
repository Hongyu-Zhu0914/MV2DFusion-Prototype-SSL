"""Build small, scene-complete nuScenes metadata roots for 2 GiB PKL conversion.

Sensor payloads are symlinked to the full dataset. The filtered metadata is
only an intermediate conversion input; it never changes the official tables.
"""

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
import json
import os
import shutil
from pathlib import Path

import ijson


SHARED_TABLES = (
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "instance.json",
    "log.json",
    "map.json",
    "sensor.json",
    "visibility.json",
)


def write_filtered(source, target, keep):
    count = 0
    with source.open("rb") as input_file, target.open("x", encoding="utf-8") as output:
        output.write("[")
        for record in ijson.items(input_file, "item", use_float=True):
            if not keep(record):
                continue
            if count:
                output.write(",")
            json.dump(record, output, ensure_ascii=False, separators=(",", ":"))
            count += 1
        output.write("]\n")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--chunk", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    full_root = args.data_root.resolve(strict=True)
    manifest_path = full_root / "ssl_splits" / "nuscenes_ssl_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.split == "train":
        scene_tokens = set(manifest["labeled_train"]["scene_tokens"])
        all_sample_tokens = set(manifest["labeled_train"]["sample_tokens"])
        expected_samples = 6000
    elif args.split == "val":
        scene_tokens = set(manifest["validation"]["scene_tokens"])
        scene_tokens.update(manifest["internal_test"]["scene_tokens"])
        all_sample_tokens = set(manifest["validation"]["sample_tokens"])
        all_sample_tokens.update(manifest["internal_test"]["sample_tokens"])
        expected_samples = 6019
    else:
        source_scene_rows = json.loads((full_root / "v1.0-test" / "scene.json").read_text())
        scene_tokens = {row["token"] for row in source_scene_rows}
        all_sample_tokens = None
        expected_samples = 6008

    source_version = "v1.0-test" if args.split == "test" else "v1.0-trainval"
    source = full_root / source_version
    scene_rows = json.loads((source / "scene.json").read_text())
    selected_rows = [row for row in scene_rows if row["token"] in scene_tokens]
    if len(selected_rows) != 150:
        raise RuntimeError(f"Expected 150 {args.split} scenes, got {len(selected_rows)}")
    buckets = [set(), set(), set()]
    bucket_lengths = [0, 0, 0]
    for row in sorted(selected_rows, key=lambda item: (-item["nbr_samples"], item["name"])):
        bucket = min(range(3), key=lambda i: (bucket_lengths[i], i))
        buckets[bucket].add(row["token"])
        bucket_lengths[bucket] += row["nbr_samples"]
    if sum(bucket_lengths) != expected_samples:
        raise RuntimeError(f"Unexpected {args.split} total: {bucket_lengths}")
    scene_tokens = buckets[args.chunk]
    sample_tokens = {
        row["token"]
        for row in json.loads((source / "sample.json").read_text())
        if row["scene_token"] in scene_tokens
    }
    if len(sample_tokens) != bucket_lengths[args.chunk]:
        raise RuntimeError("Chunk sample count differs from scene metadata")
    if all_sample_tokens is not None and not sample_tokens <= all_sample_tokens:
        raise RuntimeError("Chunk contains a sample outside the approved split")

    target_root = full_root / "pkl_generation_0924" / f"filtered_{args.split}_{args.chunk}"
    version = target_root / source_version
    version.mkdir(parents=True, exist_ok=False)
    for name in ("samples", "sweeps", "maps"):
        os.symlink(full_root / name, target_root / name, target_is_directory=True)

    counts = {}
    for name in SHARED_TABLES:
        shutil.copy2(source / name, version / name)
    counts["scene"] = write_filtered(
        source / "scene.json", version / "scene.json",
        lambda row: row["token"] in scene_tokens,
    )
    counts["sample"] = write_filtered(
        source / "sample.json", version / "sample.json",
        lambda row: row["token"] in sample_tokens,
    )
    if args.split == "test":
        shutil.copy2(source / "sample_annotation.json", version / "sample_annotation.json")
        counts["sample_annotation"] = 0
    else:
        counts["sample_annotation"] = write_filtered(
            source / "sample_annotation.json", version / "sample_annotation.json",
            lambda row: row["sample_token"] in sample_tokens,
        )
    ego_pose_tokens = set()

    def selected_sample_data(row):
        if row["sample_token"] not in sample_tokens:
            return False
        ego_pose_tokens.add(row["ego_pose_token"])
        return True

    counts["sample_data"] = write_filtered(
        source / "sample_data.json", version / "sample_data.json", selected_sample_data,
    )
    counts["ego_pose"] = write_filtered(
        source / "ego_pose.json", version / "ego_pose.json",
        lambda row: row["token"] in ego_pose_tokens,
    )
    if counts["sample"] != len(sample_tokens) or counts["scene"] != len(scene_tokens):
        raise RuntimeError(f"Incomplete {args.split} metadata: {counts}")
    if counts["ego_pose"] != len(ego_pose_tokens):
        raise RuntimeError("Some referenced ego poses are missing")
    report = {
        "split": args.split,
        "chunk": args.chunk,
        "source": str(source),
        "filtered_root": str(target_root),
        "counts": counts,
        "sensor_symlinks": ["samples", "sweeps", "maps"],
    }
    (target_root / "filter_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
