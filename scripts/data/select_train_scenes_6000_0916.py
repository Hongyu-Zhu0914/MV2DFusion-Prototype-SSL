#!/usr/bin/env python3
"""Select about 6000 labeled nuScenes train frames at whole-scene granularity.

The script builds scene-level statistics from the official nuScenes metadata,
then solves three mixed-integer programs containing 150, 151 and 152 complete
scenes respectively.  Each candidate contains exactly 6000 samples and is
optimized for class coverage, instance coverage, log diversity and geographic
coverage.  Existing log-level split files are never modified.
"""

from __future__ import annotations

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
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from nuscenes.utils.splits import create_splits_scenes


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

# These classes deserve extra protection when choosing a small labeled subset.
RARE_CLASSES = {
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "traffic_cone",
}


def iter_json_array(path: Path):
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item")


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
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name == "movable_object.trafficcone":
        return "traffic_cone"
    if category_name == "movable_object.barrier":
        return "barrier"
    return None


def build_scene_statistics(meta_root: Path):
    train_names = set(create_splits_scenes()["train"])
    logs = {row["token"]: row for row in iter_json_array(meta_root / "log.json")}
    all_scenes = list(iter_json_array(meta_root / "scene.json"))
    scenes = [row for row in all_scenes if row["name"] in train_names]
    if len(scenes) != 700:
        raise RuntimeError(f"Expected 700 official train scenes, got {len(scenes)}")

    scene_by_token = {row["token"]: row for row in scenes}
    sample_to_scene = {}
    scene_samples = defaultdict(list)
    for sample in iter_json_array(meta_root / "sample.json"):
        scene_token = sample["scene_token"]
        if scene_token in scene_by_token:
            sample_to_scene[sample["token"]] = scene_token
            scene_samples[scene_token].append(sample["token"])

    categories = {
        row["token"]: row["name"]
        for row in iter_json_array(meta_root / "category.json")
    }
    instance_class = {}
    for instance in iter_json_array(meta_root / "instance.json"):
        instance_class[instance["token"]] = detection_name(
            categories[instance["category_token"]]
        )

    stats = {}
    for scene in scenes:
        log = logs[scene["log_token"]]
        token = scene["token"]
        stats[token] = {
            "scene_token": token,
            "scene_name": scene["name"],
            "description": scene.get("description", ""),
            "log_token": scene["log_token"],
            "logfile": log["logfile"],
            "location": log["location"],
            "vehicle": log.get("vehicle", ""),
            "date_captured": log.get("date_captured", ""),
            "sample_count": len(scene_samples[token]),
            "sample_tokens": scene_samples[token],
            "boxes": Counter(),
            "class_samples": defaultdict(set),
            "instances": defaultdict(set),
        }

    annotation_path = meta_root / "sample_annotation.json"
    for ann in iter_json_array(annotation_path):
        scene_token = sample_to_scene.get(ann["sample_token"])
        if scene_token is None:
            continue
        # This mirrors the training valid_flag more closely than counting
        # completely unobserved official annotations.
        if ann.get("num_lidar_pts", 0) + ann.get("num_radar_pts", 0) <= 0:
            continue
        cls = instance_class.get(ann["instance_token"])
        if cls not in CLASSES:
            continue
        item = stats[scene_token]
        item["boxes"][cls] += 1
        item["class_samples"][cls].add(ann["sample_token"])
        item["instances"][cls].add(ann["instance_token"])

    return list(stats.values())


def totals_for(rows):
    totals = {
        "scenes": len(rows),
        "samples": sum(row["sample_count"] for row in rows),
        "logs": len({row["log_token"] for row in rows}),
        "locations": Counter(row["location"] for row in rows),
        "boxes": {},
        "class_samples": {},
        "instances": {},
        "class_scenes": {},
    }
    for cls in CLASSES:
        totals["boxes"][cls] = sum(row["boxes"][cls] for row in rows)
        totals["class_samples"][cls] = sum(
            len(row["class_samples"][cls]) for row in rows
        )
        totals["instances"][cls] = sum(len(row["instances"][cls]) for row in rows)
        totals["class_scenes"][cls] = sum(bool(row["boxes"][cls]) for row in rows)
    return totals


def solve_candidate(rows, scene_count: int, sample_target: int = 6000):
    n = len(rows)
    log_tokens = sorted({row["log_token"] for row in rows})
    log_index = {token: i for i, token in enumerate(log_tokens)}
    m = len(log_tokens)
    full = totals_for(rows)

    # Per-scene utility: equal-class normalized coverage prevents car and
    # pedestrian from drowning out genuinely rare classes.  Rare categories
    # receive a moderate additional multiplier, not an absolute takeover.
    utility = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(rows):
        for cls in CLASSES:
            rare_weight = 1.6 if cls in RARE_CLASSES else 1.0
            utility[i] += rare_weight * (
                0.50 * row["boxes"][cls] / max(full["boxes"][cls], 1)
                + 0.35 * len(row["instances"][cls]) / max(full["instances"][cls], 1)
                + 0.15
                * len(row["class_samples"][cls])
                / max(full["class_samples"][cls], 1)
            )
        # Deterministic, negligible tie-breaker.
        utility[i] += (n - i) * 1e-10

    # Variables: scene selections x, plus one log-covered indicator y per log.
    var_count = n + m
    c = np.zeros(var_count, dtype=np.float64)
    c[:n] = -utility
    c[n:] = -0.018  # up to 0.9 utility for covering all 50 train logs

    constraints = []
    lower = []
    upper = []

    def add(coefficients, lb, ub):
        constraints.append(coefficients)
        lower.append(lb)
        upper.append(ub)

    coeff = np.zeros(var_count)
    coeff[:n] = 1
    add(coeff, scene_count, scene_count)

    coeff = np.zeros(var_count)
    coeff[:n] = [row["sample_count"] for row in rows]
    add(coeff, sample_target, sample_target)

    # Link each covered-log flag to the selected scenes from that log.
    for token, j in log_index.items():
        coeff = np.zeros(var_count)
        coeff[n + j] = 1
        for i, row in enumerate(rows):
            if row["log_token"] == token:
                coeff[i] -= 1
        add(coeff, -np.inf, 0)

        # Avoid letting a single long log dominate the labeled subset.
        coeff = np.zeros(var_count)
        for i, row in enumerate(rows):
            if row["log_token"] == token:
                coeff[i] = 1
        add(coeff, -np.inf, 7)

    coeff = np.zeros(var_count)
    coeff[n:] = 1
    add(coeff, 45, np.inf)

    # Preserve each geographic domain near its full-train proportion.
    locations = sorted(full["locations"])
    for location in locations:
        expected = scene_count * full["locations"][location] / n
        loc_min = max(1, int(np.floor(0.72 * expected)))
        loc_max = int(np.ceil(1.35 * expected))
        coeff = np.zeros(var_count)
        for i, row in enumerate(rows):
            if row["location"] == location:
                coeff[i] = 1
        add(coeff, loc_min, loc_max)

    # A 6000/28130-frame split is 21.3% of train.  These conservative lower
    # bounds stop the optimizer from sacrificing a class for another class.
    for cls in CLASSES:
        min_fraction = 0.18 if cls in RARE_CLASSES else 0.15
        coeff = np.zeros(var_count)
        coeff[:n] = [row["boxes"][cls] for row in rows]
        add(coeff, int(np.floor(min_fraction * full["boxes"][cls])), np.inf)

        coeff = np.zeros(var_count)
        coeff[:n] = [len(row["instances"][cls]) for row in rows]
        add(coeff, int(np.floor(0.15 * full["instances"][cls])), np.inf)

    matrix = lil_matrix((len(constraints), var_count), dtype=np.float64)
    for r, values in enumerate(constraints):
        nz = np.flatnonzero(values)
        matrix[r, nz] = values[nz]

    result = milp(
        c=c,
        integrality=np.ones(var_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(var_count), np.ones(var_count)),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lower), np.array(upper)),
        options={"time_limit": 300, "mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success:
        raise RuntimeError(
            f"MILP failed for {scene_count} scenes: status={result.status}, "
            f"message={result.message}"
        )
    selected = [rows[i] for i in range(n) if result.x[i] > 0.5]
    if len(selected) != scene_count:
        raise RuntimeError("MILP returned a non-integral scene selection")
    return sorted(selected, key=lambda row: row["scene_name"]), float(-result.fun)


def serialize_totals(summary, full):
    result = {
        "scene_count": summary["scenes"],
        "sample_count": summary["samples"],
        "log_count": summary["logs"],
        "location_scene_counts": dict(summary["locations"]),
        "classes": {},
    }
    for cls in CLASSES:
        result["classes"][cls] = {
            "boxes": summary["boxes"][cls],
            "box_retention_pct": 100.0
            * summary["boxes"][cls]
            / max(full["boxes"][cls], 1),
            "instances": summary["instances"][cls],
            "instance_retention_pct": 100.0
            * summary["instances"][cls]
            / max(full["instances"][cls], 1),
            "positive_samples": summary["class_samples"][cls],
            "positive_sample_retention_pct": 100.0
            * summary["class_samples"][cls]
            / max(full["class_samples"][cls], 1),
            "positive_scenes": summary["class_scenes"][cls],
        }
    return result


def write_scene_csv(path: Path, rows):
    fields = [
        "scene_name",
        "scene_token",
        "sample_count",
        "logfile",
        "log_token",
        "location",
        "vehicle",
        "date_captured",
        "total_valid_boxes",
    ]
    for cls in CLASSES:
        fields.extend(
            [f"{cls}_boxes", f"{cls}_instances", f"{cls}_positive_samples"]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in fields[:8]}
            out["total_valid_boxes"] = sum(row["boxes"].values())
            for cls in CLASSES:
                out[f"{cls}_boxes"] = row["boxes"][cls]
                out[f"{cls}_instances"] = len(row["instances"][cls])
                out[f"{cls}_positive_samples"] = len(row["class_samples"][cls])
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=Path("/root/autodl-tmp/data/nuscenes/v1.0-trainval"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/data/nuscenes/scene_selection_6000_0916"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_scene_statistics(args.meta_root)
    rows.sort(key=lambda row: row["scene_name"])
    full = totals_for(rows)
    write_scene_csv(args.out_dir / "train_scene_statistics_all700.csv", rows)

    result = {
        "method": "whole-scene MILP selection with valid-point annotations",
        "target_samples": 6000,
        "unlabeled_test_reference_samples": 6008,
        "full_train": serialize_totals(full, full),
        "candidates": {},
    }
    candidates = {}
    for count in (150, 151, 152):
        selected, objective = solve_candidate(rows, count, 6000)
        name = f"balanced_{count}scenes_6000samples"
        candidates[name] = selected
        write_scene_csv(args.out_dir / f"{name}.csv", selected)
        (args.out_dir / f"{name}_scene_tokens.txt").write_text(
            "\n".join(row["scene_token"] for row in selected) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / f"{name}_sample_tokens.txt").write_text(
            "\n".join(token for row in selected for token in row["sample_tokens"])
            + "\n",
            encoding="utf-8",
        )
        summary = totals_for(selected)
        candidate_result = serialize_totals(summary, full)
        candidate_result["solver_objective"] = objective
        candidate_result["sample_ratio_vs_unlabeled_test_pct"] = (
            100.0 * summary["samples"] / 6008
        )
        candidate_result["scene_length_histogram"] = dict(
            sorted(Counter(row["sample_count"] for row in selected).items())
        )
        candidate_result["log_scene_counts"] = dict(
            sorted(Counter(row["logfile"] for row in selected).items())
        )
        result["candidates"][name] = candidate_result

    # Primary recommendation: maximizing scene count is only useful if it does
    # not materially damage the categories for which labeled data are scarcest.
    # Compare the macro-average retention of rare-class boxes; this deliberately
    # does not let the abundant car class conceal rare-category losses.
    def rare_box_macro(candidate):
        return float(
            np.mean(
                [
                    result["candidates"][candidate]["classes"][cls][
                        "box_retention_pct"
                    ]
                    for cls in sorted(RARE_CLASSES)
                ]
            )
        )

    strongest_name = max(candidates, key=rare_box_macro)
    max_scene_name = "balanced_152scenes_6000samples"
    rare_loss = rare_box_macro(strongest_name) - rare_box_macro(max_scene_name)
    if rare_loss <= 2.0:
        recommended = max_scene_name
        reason = (
            "Maximum number of complete scenes at 6000 samples, with no material "
            "(>2 percentage-point) loss in macro rare-class box retention."
        )
    else:
        recommended = strongest_name
        reason = (
            f"Chosen over the 152-scene maximum because the extra two scenes cost "
            f"{rare_loss:.2f} percentage points of macro rare-class box retention."
        )
    for candidate in candidates:
        result["candidates"][candidate]["rare_box_macro_retention_pct"] = (
            rare_box_macro(candidate)
        )
    result["recommended_candidate"] = recommended
    result["recommendation_reason"] = reason

    summary_path = args.out_dir / "selection_comparison.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote selection artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
