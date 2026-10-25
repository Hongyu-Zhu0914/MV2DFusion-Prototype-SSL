"""Assemble scene-complete chunk conversions into the four approved SSL PKLs."""

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path


PARTIAL_NAME = {
    "train": "nuscenes2d_temporal_infos_train.pkl",
    "val": "nuscenes2d_temporal_infos_val.pkl",
    "test": "nuscenes2d_temporal_infos_test.pkl",
}
EXPECTED = {"train": 6000, "val": 6019, "test": 6008}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chunks(data_root, split):
    root = data_root / "pkl_generation_0924"
    infos = []
    scenes = set()
    for index in range(3):
        directory = root / f"filtered_{split}_{index}"
        report = json.loads((directory / "filter_report.json").read_text())
        path = directory / PARTIAL_NAME[split]
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        part = payload["infos"]
        if len(part) != report["counts"]["sample"]:
            raise RuntimeError(f"Chunk count mismatch: {path}")
        if payload["metadata"]["version"] != ("v1.0-test" if split == "test" else "v1.0-trainval"):
            raise RuntimeError(f"Unexpected metadata version: {path}")
        infos.extend(part)
        scenes.update(row["scene_token"] for row in part)
        print(f"LOADED {split} chunk={index} count={len(part)}", flush=True)
    infos.sort(key=lambda row: (row["timestamp"], row["token"]))
    tokens = [row["token"] for row in infos]
    if len(tokens) != len(set(tokens)) or len(tokens) != EXPECTED[split]:
        raise RuntimeError(f"Missing or duplicate {split} records: {len(tokens)}")
    if len(scenes) != 150:
        raise RuntimeError(f"Expected 150 {split} scenes, got {len(scenes)}")
    return infos


def check_infos(infos, data_root, expect_gt):
    missing = []
    classes = Counter()
    path_count = 0
    for row in infos:
        has_gt = all(key in row for key in ("gt_boxes", "gt_names", "valid_flag"))
        if has_gt != expect_gt:
            raise RuntimeError(f"GT presence mismatch for {row['token']}")
        if expect_gt:
            for key in ("bboxes2d", "labels2d", "bboxes3d_cams", "centers2d", "depths"):
                if key not in row or len(row[key]) != 6:
                    raise RuntimeError(f"Missing 2D supervision: {key} {row['token']}")
            classes.update(str(x) for x, valid in zip(row["gt_names"], row["valid_flag"]) if valid)
        paths = [row["lidar_path"]]
        paths.extend(cam["data_path"] for cam in row["cams"].values())
        paths.extend(sweep["data_path"] for sweep in row.get("sweeps", []))
        for value in paths:
            path = Path(value)
            if not path.is_absolute():
                path = data_root / path
            path_count += 1
            if not path.is_file() and len(missing) < 10:
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"Sensor paths missing: {missing}")
    return {"checked_sensor_paths": path_count, "valid_boxes_by_class": dict(classes)}


def write_pickle(target, infos, version):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as handle:
        pickle.dump({"infos": infos, "metadata": {"version": version}}, handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(target)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target), "frames": len(infos)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    manifest = json.loads((root / "ssl_splits" / "nuscenes_ssl_split_manifest.json").read_text())
    infos = load_chunks(root, args.split)
    checks = check_infos(infos, root, expect_gt=args.split != "test")
    outputs = []
    if args.split == "train":
        expected = set(manifest["labeled_train"]["sample_tokens"])
        if {row["token"] for row in infos} != expected:
            raise RuntimeError("Labeled train tokens differ from approved manifest")
        outputs.append(write_pickle(
            root / "ssl_splits" / "nuscenes_ssl_train_labeled_150scenes_6000_0919.pkl",
            infos, "v1.0-trainval",
        ))
    elif args.split == "val":
        expected_val = set(manifest["validation"]["sample_tokens"])
        expected_test = set(manifest["internal_test"]["sample_tokens"])
        if {row["token"] for row in infos} != expected_val | expected_test:
            raise RuntimeError("Official val tokens differ from approved manifest")
        outputs.append(write_pickle(root / PARTIAL_NAME["val"], infos, "v1.0-trainval"))
        for filename, tokens in (
            ("nuscenes_ssl_val_split.pkl", expected_val),
            ("nuscenes_ssl_test_split.pkl", expected_test),
        ):
            subset = [row for row in infos if row["token"] in tokens]
            if len(subset) != len(tokens):
                raise RuntimeError(f"Incomplete {filename}")
            outputs.append(write_pickle(root / "ssl_splits" / filename, subset, "v1.0-trainval"))
    else:
        outputs.append(write_pickle(root / PARTIAL_NAME["test"], infos, "v1.0-test"))
        outputs.append(write_pickle(
            root / "ssl_splits" / "nuscenes_ssl_train_unlabeled_official_test.pkl",
            infos, "v1.0-test",
        ))
    report = {"split": args.split, "checks": checks, "outputs": outputs}
    report_path = root / "pkl_generation_0924" / f"assembled_{args.split}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
