#!/usr/bin/env python3
"""Generate reproducible MV2DFusion semi-supervised nuScenes PKLs.

The large protocol-2 train pickle is filtered with a memory-pruned pure-Python
unpickler. Selected info dictionaries are spooled as independent pickles, then
written into one normal ``{'infos': [...], 'metadata': ...}`` pickle through a
streaming list pickler. This avoids materializing all 28k train infos at once.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import pickle
import pickletools
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple


CLASSES = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_selected_logfiles(csv_path: Path) -> List[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        values = [row["logfile"] for row in csv.DictReader(handle)]
    if len(values) != 15 or len(set(values)) != 15:
        raise RuntimeError(f"Expected 15 unique labeled logs, got {len(values)}")
    return values


def metadata_indices(data_root: Path):
    version_root = data_root / "v1.0-trainval"
    logs = {x["token"]: x for x in load_json(version_root / "log.json")}
    scenes = {x["token"]: x for x in load_json(version_root / "scene.json")}
    samples = load_json(version_root / "sample.json")
    sample_to_scene = {x["token"]: x["scene_token"] for x in samples}
    return version_root, logs, scenes, samples, sample_to_scene


def selected_train_tokens(
    logfiles: Sequence[str], logs: Dict, scenes: Dict, samples: Sequence[Dict]
) -> Tuple[Set[str], Set[str]]:
    log_tokens = {token for token, row in logs.items() if row["logfile"] in logfiles}
    if len(log_tokens) != len(logfiles):
        found = {logs[token]["logfile"] for token in log_tokens}
        raise RuntimeError(f"Missing labeled logfiles: {sorted(set(logfiles) - found)}")
    scene_tokens = {token for token, row in scenes.items() if row["log_token"] in log_tokens}
    sample_tokens = {row["token"] for row in samples if row["scene_token"] in scene_tokens}
    return scene_tokens, sample_tokens


def detection_name(category_name: str):
    if category_name == "vehicle.car":
        return "car"
    if category_name == "vehicle.truck":
        return "truck"
    if category_name == "vehicle.trailer":
        return "trailer"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name == "vehicle.construction":
        return "construction_vehicle"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name in {
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    }:
        return "pedestrian"
    if category_name == "movable_object.trafficcone":
        return "traffic_cone"
    if category_name == "movable_object.barrier":
        return "barrier"
    return None


def build_val_log_stats(
    version_root: Path,
    logs: Dict,
    scenes: Dict,
    samples: Sequence[Dict],
):
    import ijson
    from nuscenes.utils.splits import create_splits_scenes

    val_names = set(create_splits_scenes()["val"])
    val_scenes = {token: row for token, row in scenes.items() if row["name"] in val_names}
    if len(val_scenes) != 150:
        raise RuntimeError(f"Expected 150 val scenes, got {len(val_scenes)}")

    sample_to_log = {}
    stats = {}
    for token, row in val_scenes.items():
        log_token = row["log_token"]
        if log_token not in stats:
            stats[log_token] = {
                "log_token": log_token,
                "logfile": logs[log_token]["logfile"],
                "location": logs[log_token]["location"],
                "scene_tokens": [],
                "sample_count": 0,
                "boxes": Counter(),
            }
        stats[log_token]["scene_tokens"].append(token)
    for sample in samples:
        scene_token = sample["scene_token"]
        if scene_token in val_scenes:
            log_token = val_scenes[scene_token]["log_token"]
            sample_to_log[sample["token"]] = log_token
            stats[log_token]["sample_count"] += 1

    categories = {x["token"]: x["name"] for x in load_json(version_root / "category.json")}
    instances = {
        x["token"]: detection_name(categories[x["category_token"]])
        for x in load_json(version_root / "instance.json")
    }
    with (version_root / "sample_annotation.json").open("rb") as handle:
        for ann in ijson.items(handle, "item"):
            log_token = sample_to_log.get(ann["sample_token"])
            if log_token is None:
                continue
            name = instances.get(ann["instance_token"])
            if name in CLASSES:
                stats[log_token]["boxes"][name] += 1
    return stats


def choose_balanced_val_logs(stats: Dict) -> Tuple[List[str], List[str]]:
    rows = sorted(stats.values(), key=lambda x: x["logfile"])
    totals = {c: sum(r["boxes"][c] for r in rows) for c in CLASSES}
    total_samples = sum(r["sample_count"] for r in rows)
    total_location_scenes = Counter()
    for row in rows:
        total_location_scenes[row["location"]] += len(row["scene_tokens"])

    best = None
    best_cost = math.inf
    # Complementary assignments are equivalent. Pin the first log to val.
    for mask_tail in range(1 << (len(rows) - 1)):
        ids = [0]
        ids.extend(i + 1 for i in range(len(rows) - 1) if mask_tail & (1 << i))
        scene_count = sum(len(rows[i]["scene_tokens"]) for i in ids)
        if abs(scene_count - 75) > 2:
            continue
        sample_count = sum(rows[i]["sample_count"] for i in ids)
        cost = 8.0 * abs(scene_count - 75) / 75.0
        cost += 1.5 * abs(sample_count / total_samples - 0.5)
        for c in CLASSES:
            if totals[c]:
                ratio = sum(rows[i]["boxes"][c] for i in ids) / totals[c]
                cost += abs(ratio - 0.5)
        loc_scenes = Counter()
        for i in ids:
            loc_scenes[rows[i]["location"]] += len(rows[i]["scene_tokens"])
        for loc, total in total_location_scenes.items():
            cost += 0.75 * abs(loc_scenes[loc] / total - 0.5)
        if cost < best_cost:
            best_cost = cost
            best = ids
    if best is None:
        raise RuntimeError("Could not find a near-75/75 whole-log val split")
    val_logs = [rows[i]["log_token"] for i in best]
    test_logs = [row["log_token"] for i, row in enumerate(rows) if i not in set(best)]
    return val_logs, test_logs


def collect_get_memo_ids(path: Path) -> Set[int]:
    used = set()
    with path.open("rb") as handle:
        for op, arg, _ in pickletools.genops(handle):
            if op.name in {"BINGET", "LONG_BINGET"}:
                used.add(int(arg))
            elif op.name == "GET":
                used.add(int(arg))
    return used


class FilterSpoolUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file, keep_tokens: Set[str], spool_handle, used_memo: Set[int]):
        super().__init__(file, fix_imports=True, encoding="latin1")
        self.keep_tokens = keep_tokens
        self.spool_handle = spool_handle
        self.used_memo = used_memo
        self.info_list = None
        self.total = 0
        self.kept = 0

    def load_empty_list(self):
        value = []
        self.append(value)
        if self.info_list is None:
            self.info_list = value

    def _process_info(self, item):
        self.total += 1
        if item.get("token") in self.keep_tokens:
            pickle.dump(item, self.spool_handle, protocol=4)
            self.kept += 1

    def load_append(self):
        value = self.stack.pop()
        target = self.stack[-1]
        if target is self.info_list and isinstance(value, dict) and "token" in value:
            self._process_info(value)
        else:
            target.append(value)

    def load_appends(self):
        items = self.pop_mark()
        target = self.stack[-1]
        if target is self.info_list:
            for item in items:
                self._process_info(item)
        else:
            target.extend(items)

    def load_binput(self):
        index = self.read(1)[0]
        if index in self.used_memo:
            self.memo[index] = self.stack[-1]

    def load_long_binput(self):
        index, = struct.unpack("<I", self.read(4))
        if index in self.used_memo:
            self.memo[index] = self.stack[-1]


FilterSpoolUnpickler.dispatch[pickle.EMPTY_LIST[0]] = FilterSpoolUnpickler.load_empty_list
FilterSpoolUnpickler.dispatch[pickle.APPEND[0]] = FilterSpoolUnpickler.load_append
FilterSpoolUnpickler.dispatch[pickle.APPENDS[0]] = FilterSpoolUnpickler.load_appends
FilterSpoolUnpickler.dispatch[pickle.BINPUT[0]] = FilterSpoolUnpickler.load_binput
FilterSpoolUnpickler.dispatch[pickle.LONG_BINPUT[0]] = FilterSpoolUnpickler.load_long_binput


class SpoolList:
    def __init__(self, path: Path, count: int):
        self.path = path
        self.count = count

    def __iter__(self) -> Iterator[Dict]:
        with self.path.open("rb") as handle:
            for _ in range(self.count):
                yield pickle.load(handle)


class StreamingPickler(pickle._Pickler):
    dispatch = pickle._Pickler.dispatch.copy()

    def save_spool_list(self, obj: SpoolList):
        # The records in the spool are independent.  Writing them through
        # _batch_appends would keep every record in the Pickler memo until the
        # complete 13k-item list is finished, which exceeds the 2 GiB cgroup.
        # Protocol 2 uses explicit BINPUT indices, so clearing the memo after a
        # completed item is safe: a later record can overwrite those indices
        # and no record contains a reference to an earlier one.
        if self.bin:
            self.write(pickle.EMPTY_LIST)
        else:
            self.write(pickle.MARK + pickle.LIST)
        for item in obj:
            self.save(item)
            self.write(pickle.APPEND)
            self.clear_memo()


StreamingPickler.dispatch[SpoolList] = StreamingPickler.save_spool_list


def filter_large_pickle(source: Path, target: Path, keep_tokens: Set[str], work_dir: Path):
    memo_cache = work_dir / (source.name + ".used_memo.json")
    if memo_cache.exists():
        used_memo = set(load_json(memo_cache))
    else:
        used_memo = collect_get_memo_ids(source)
        memo_cache.write_text(json.dumps(sorted(used_memo)), encoding="utf-8")
    spool = work_dir / (target.name + ".records.tmp")
    with source.open("rb") as src, spool.open("wb") as spool_handle:
        unpickler = FilterSpoolUnpickler(src, keep_tokens, spool_handle, used_memo)
        data = unpickler.load()
        total, kept = unpickler.total, unpickler.kept
    if kept != len(keep_tokens):
        raise RuntimeError(f"Expected {len(keep_tokens)} selected infos, wrote {kept}; source had {total}")
    metadata = data.get("metadata", {})
    del data
    gc.collect()
    partial = target.with_suffix(target.suffix + ".partial")
    with partial.open("wb") as handle:
        pickler = StreamingPickler(handle, protocol=2)
        pickler.dump({"infos": SpoolList(spool, kept), "metadata": metadata})
    partial.replace(target)
    spool.unlink()
    return {"source_infos": total, "output_infos": kept, "metadata": metadata}


def dump_filtered_loaded(source: Path, outputs: Sequence[Tuple[Path, Set[str]]]):
    with source.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    all_tokens = {x["token"] for x in data["infos"]}
    results = []
    for target, tokens in outputs:
        infos = [x for x in data["infos"] if x["token"] in tokens]
        if len(infos) != len(tokens):
            raise RuntimeError(f"{target.name}: expected {len(tokens)} infos, got {len(infos)}")
        partial = target.with_suffix(target.suffix + ".partial")
        with partial.open("wb") as handle:
            pickle.dump({"infos": infos, "metadata": data.get("metadata", {})}, handle, protocol=4)
        partial.replace(target)
        results.append({"path": str(target), "infos": len(infos)})
    if set().union(*(tokens for _, tokens in outputs)) != all_tokens:
        raise RuntimeError("Val/test outputs do not exactly partition source val tokens")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["plan", "train", "assemble", "val", "test", "all"],
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes"))
    parser.add_argument("--test-root", type=Path, default=Path("/root/autodl-tmp/data/nuscenes_test"))
    parser.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/data/nuscenes/ssl_splits"))
    parser.add_argument(
        "--selected-logs",
        type=Path,
        default=Path("/root/autodl-tmp/data/nuscenes/train_log_statistics_recommended15_smallclass_scene_rich.csv"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / ".work"
    work_dir.mkdir(exist_ok=True)
    manifest_path = args.out_dir / "nuscenes_ssl_split_manifest.json"

    logfiles = load_selected_logfiles(args.selected_logs)
    version_root, logs, scenes, samples, sample_to_scene = metadata_indices(args.data_root)
    train_scenes, train_tokens = selected_train_tokens(logfiles, logs, scenes, samples)

    if args.stage in {"plan", "all"} or not manifest_path.exists():
        val_stats = build_val_log_stats(version_root, logs, scenes, samples)
        val_log_tokens, test_log_tokens = choose_balanced_val_logs(val_stats)
        val_scene_tokens = sorted(
            token for token, row in scenes.items() if row["log_token"] in set(val_log_tokens)
        )
        test_scene_tokens = sorted(
            token for token, row in scenes.items() if row["log_token"] in set(test_log_tokens)
        )
        val_tokens = sorted(row["token"] for row in samples if row["scene_token"] in set(val_scene_tokens))
        test_tokens = sorted(row["token"] for row in samples if row["scene_token"] in set(test_scene_tokens))
        manifest = {
            "description": "MV2DFusion SSL split: 15 labeled train logs, official test unlabeled, official val split by whole log",
            "labeled_train": {
                "logfiles": sorted(logfiles),
                "scene_tokens": sorted(train_scenes),
                "sample_tokens": sorted(train_tokens),
            },
            "unlabeled_train": {
                "source": "official nuScenes v1.0-test",
                "expected_scenes": 150,
                "expected_samples": 6008,
            },
            "validation": {
                "log_tokens": sorted(val_log_tokens),
                "logfiles": sorted(logs[x]["logfile"] for x in val_log_tokens),
                "scene_tokens": val_scene_tokens,
                "sample_tokens": val_tokens,
            },
            "internal_test": {
                "log_tokens": sorted(test_log_tokens),
                "logfiles": sorted(logs[x]["logfile"] for x in test_log_tokens),
                "scene_tokens": test_scene_tokens,
                "sample_tokens": test_tokens,
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PLAN {manifest_path}")
        print(
            f"labeled={len(train_scenes)} scenes/{len(train_tokens)} samples, "
            f"val={len(val_scene_tokens)} scenes/{len(val_tokens)} samples, "
            f"test={len(test_scene_tokens)} scenes/{len(test_tokens)} samples"
        )
    else:
        manifest = load_json(manifest_path)

    del logs, scenes, samples, sample_to_scene
    gc.collect()

    if args.stage in {"train", "all"}:
        result = filter_large_pickle(
            args.data_root / "nuscenes2d_temporal_infos_train.pkl",
            args.out_dir / "nuscenes_ssl_train_labeled_15log.pkl",
            set(manifest["labeled_train"]["sample_tokens"]),
            work_dir,
        )
        print("TRAIN", json.dumps(result, ensure_ascii=False))

    if args.stage == "assemble":
        target = args.out_dir / "nuscenes_ssl_train_labeled_15log.pkl"
        spool = work_dir / (target.name + ".records.tmp")
        if not spool.exists():
            raise FileNotFoundError(f"Missing completed record spool: {spool}")
        count = len(manifest["labeled_train"]["sample_tokens"])
        partial = target.with_suffix(target.suffix + ".partial")
        with partial.open("wb") as handle:
            StreamingPickler(handle, protocol=2).dump(
                {
                    "infos": SpoolList(spool, count),
                    "metadata": {"version": "v1.0-trainval"},
                }
            )
        partial.replace(target)
        print("ASSEMBLE", target, count)

    if args.stage in {"val", "all"}:
        result = dump_filtered_loaded(
            args.data_root / "nuscenes2d_temporal_infos_val.pkl",
            [
                (
                    args.out_dir / "nuscenes_ssl_val_split.pkl",
                    set(manifest["validation"]["sample_tokens"]),
                ),
                (
                    args.out_dir / "nuscenes_ssl_test_split.pkl",
                    set(manifest["internal_test"]["sample_tokens"]),
                ),
            ],
        )
        print("VAL_TEST", json.dumps(result, ensure_ascii=False))

    if args.stage in {"test", "all"}:
        source = args.test_root / "nuscenes2d_temporal_infos_test.pkl"
        target = args.out_dir / "nuscenes_ssl_train_unlabeled_official_test.pkl"
        partial = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(source, partial)
        partial.replace(target)
        print("UNLABELED_TEST", target, source.stat().st_size)

    generated = [p for p in args.out_dir.glob("*.pkl") if p.is_file()]
    if generated:
        hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in generated}
        (args.out_dir / "nuscenes_ssl_pkl_hashes.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8"
        )
        print("HASHES", args.out_dir / "nuscenes_ssl_pkl_hashes.json")


if __name__ == "__main__":
    main()
