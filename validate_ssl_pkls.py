#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path

from generate_ssl_pkls import FilterSpoolUnpickler, collect_get_memo_ids


GT_KEYS = ("gt_boxes", "gt_names")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(value, repo_root: Path):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    value = value[2:] if value.startswith("./") else value
    return repo_root / value


def inspect_info(info, tokens, scenes, require_gt, repo_root, missing_paths):
    token = info.get("token")
    if not token:
        raise RuntimeError("An info record has no token")
    if token in tokens:
        raise RuntimeError(f"Duplicate sample token: {token}")
    tokens.add(token)
    if info.get("scene_token"):
        scenes.add(info["scene_token"])
    has_gt = all(key in info for key in GT_KEYS)
    if require_gt is True and not has_gt:
        raise RuntimeError(f"Labeled info has no GT: {token}")
    if require_gt is False and any(key in info for key in GT_KEYS):
        raise RuntimeError(f"Unlabeled info unexpectedly has GT: {token}")
    paths = [info.get("lidar_path")]
    paths.extend(cam.get("data_path") for cam in info.get("cams", {}).values())
    for value in paths:
        path = resolve_data_path(value, repo_root)
        if path is not None and not path.exists() and len(missing_paths) < 20:
            missing_paths.append(str(path))


def inspect_spool(path, expected_count, repo_root):
    tokens, scenes, missing_paths = set(), set(), []
    with path.open("rb") as handle:
        for _ in range(expected_count):
            try:
                info = pickle.load(handle)
            except EOFError as exc:
                raise RuntimeError("Spool ended before expected record count") from exc
            inspect_info(info, tokens, scenes, True, repo_root, missing_paths)
        try:
            pickle.load(handle)
        except EOFError:
            pass
        else:
            raise RuntimeError("Spool contains more records than expected")
    return tokens, scenes, missing_paths


def validate_final_train_stream(path, expected_count):
    used_memo = collect_get_memo_ids(path)
    with path.open("rb") as source, open(os.devnull, "wb") as sink:
        loader = FilterSpoolUnpickler(source, set(), sink, used_memo)
        data = loader.load()
        trailing = source.read(1)
    if loader.total != expected_count or loader.kept != 0:
        raise RuntimeError(
            f"Final train stream count mismatch: total={loader.total}, kept={loader.kept}"
        )
    if trailing:
        raise RuntimeError("Final train pickle has trailing bytes")
    return data.get("metadata", {})


def inspect_normal_pickle(path, require_gt, repo_root):
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    tokens, scenes, missing_paths = set(), set(), []
    for info in data["infos"]:
        inspect_info(info, tokens, scenes, require_gt, repo_root, missing_paths)
    metadata = data.get("metadata", {})
    del data
    gc.collect()
    return tokens, scenes, missing_paths, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("/root/autodl-tmp/data/nuscenes/ssl_splits"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/root/MV2DFusion-main"))
    args = parser.parse_args()
    out = args.out_dir
    manifest = json.loads((out / "nuscenes_ssl_split_manifest.json").read_text())
    expected_train = set(manifest["labeled_train"]["sample_tokens"])
    expected_val = set(manifest["validation"]["sample_tokens"])
    expected_internal_test = set(manifest["internal_test"]["sample_tokens"])

    train_path = out / "nuscenes_ssl_train_labeled_15log.pkl"
    spool_path = out / ".work" / (train_path.name + ".records.tmp")
    train_tokens, train_scenes, train_missing = inspect_spool(
        spool_path, len(expected_train), args.repo_root
    )
    if train_tokens != expected_train:
        raise RuntimeError("Labeled train tokens do not exactly match manifest")
    if train_scenes != set(manifest["labeled_train"]["scene_tokens"]):
        raise RuntimeError("Labeled train scenes do not exactly match manifest")
    train_metadata = validate_final_train_stream(train_path, len(expected_train))

    val_path = out / "nuscenes_ssl_val_split.pkl"
    val_tokens, val_scenes, val_missing, val_metadata = inspect_normal_pickle(
        val_path, True, args.repo_root
    )
    if val_tokens != expected_val or val_scenes != set(manifest["validation"]["scene_tokens"]):
        raise RuntimeError("Validation split does not exactly match manifest")

    internal_test_path = out / "nuscenes_ssl_test_split.pkl"
    it_tokens, it_scenes, it_missing, it_metadata = inspect_normal_pickle(
        internal_test_path, True, args.repo_root
    )
    if it_tokens != expected_internal_test or it_scenes != set(manifest["internal_test"]["scene_tokens"]):
        raise RuntimeError("Internal test split does not exactly match manifest")

    unlabeled_path = out / "nuscenes_ssl_train_unlabeled_official_test.pkl"
    ul_tokens, ul_scenes, ul_missing, ul_metadata = inspect_normal_pickle(
        unlabeled_path, False, args.repo_root
    )
    if len(ul_tokens) != 6008 or len(ul_scenes) != 150:
        raise RuntimeError("Official test split is not 6008 samples / 150 scenes")

    if val_tokens & it_tokens or val_scenes & it_scenes:
        raise RuntimeError("Validation and internal test splits overlap")
    if train_tokens & (val_tokens | it_tokens) or train_scenes & (val_scenes | it_scenes):
        raise RuntimeError("Labeled train overlaps with validation/internal test")

    files = [train_path, unlabeled_path, val_path, internal_test_path]
    report = {
        "status": "PASS",
        "labeled_train": {
            "samples": len(train_tokens), "scenes": len(train_scenes),
            "gt": "present", "metadata": train_metadata,
            "missing_primary_paths": train_missing,
        },
        "unlabeled_official_test": {
            "samples": len(ul_tokens), "scenes": len(ul_scenes),
            "gt": "absent", "metadata": ul_metadata,
            "missing_primary_paths": ul_missing,
        },
        "validation": {
            "samples": len(val_tokens), "scenes": len(val_scenes),
            "gt": "present", "metadata": val_metadata,
            "missing_primary_paths": val_missing,
        },
        "internal_test": {
            "samples": len(it_tokens), "scenes": len(it_scenes),
            "gt": "present", "metadata": it_metadata,
            "missing_primary_paths": it_missing,
        },
        "disjoint": {
            "validation_vs_internal_test": True,
            "labeled_train_vs_val_and_test": True,
        },
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    missing = train_missing + ul_missing + val_missing + it_missing
    if missing:
        raise RuntimeError(f"Missing referenced data paths, examples: {missing[:5]}")
    report_path = out / "nuscenes_ssl_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPORT", report_path)


if __name__ == "__main__":
    main()
