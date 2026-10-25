import argparse
import csv
import math
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import Polygon


CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
ACTIVE_CLASSES = [0, 1, 3, 6, 7, 8, 9]
DISTANCE_THRESHOLDS = {
    0: 1.0,
    1: 1.2,
    3: 1.5,
    6: 0.5,
    7: 0.4,
    8: 0.35,
    9: 0.25,
}
DIRECTIONAL_CLASSES = {0, 1, 3, 6, 7}
COLORS = {
    "positive": "#4C78A8",
    "negative": "#E45756",
    "overlap": "#59A14F",
    "error": "#E15759",
    "count": "#D9D9D9",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze detector and prototype behavior by GT-to-ego BEV "
            "distance for the seven active nuScenes classes."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--proto-file", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="checkpoint")
    parser.add_argument("--positive-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--positive-angle-threshold-deg", type=float, default=45.0)
    parser.add_argument("--clear-negative-distance-threshold", type=float, default=2.0)
    parser.add_argument(
        "--distance-edges",
        default="0,10,20,30,40,50,inf",
        help="Comma-separated GT-to-ego BEV distance bin edges in meters.",
    )
    return parser.parse_args()


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def normalize(values):
    return values / (
        np.linalg.norm(values, axis=-1, keepdims=True) + 1e-8
    )


def parse_edges(value):
    edges = []
    for item in value.split(","):
        item = item.strip().lower()
        edges.append(float("inf") if item == "inf" else float(item))
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) < 2 or edges[0] != 0 or not np.all(np.diff(edges) > 0):
        raise ValueError("distance edges must start at 0 and be increasing")
    return edges


def bin_labels(edges):
    labels = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if np.isinf(upper):
            labels.append(f"{lower:g}+")
        else:
            labels.append(f"{lower:g}-{upper:g}")
    return labels


def distance_bin(distance, edges):
    index = int(np.searchsorted(edges, distance, side="right") - 1)
    return max(0, min(index, len(edges) - 2))


def angle_difference(first, second):
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def load_prototypes(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "prototypes" in checkpoint:
        prototypes = checkpoint["prototypes"]
    else:
        prototypes = None
        for key, value in checkpoint.get("state_dict", {}).items():
            if key.endswith(".prototypes"):
                prototypes = value
                break
        if prototypes is None:
            raise RuntimeError(f"Cannot find prototypes in {path}")
    return normalize(
        prototypes.detach().float().cpu().numpy().astype(np.float32)
    )


def get_bev_corners(box):
    center_x, center_y = float(box[0]), float(box[1])
    width, length = float(box[3]), float(box[4])
    yaw = float(box[6])
    corners = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float32,
    )
    return corners @ rotation.T + np.array(
        [center_x, center_y],
        dtype=np.float32,
    )


def overlap_over_gt_area(prediction_box, ground_truth_box):
    prediction_polygon = Polygon(get_bev_corners(prediction_box))
    ground_truth_polygon = Polygon(get_bev_corners(ground_truth_box))
    if (
        not prediction_polygon.is_valid
        or not ground_truth_polygon.is_valid
        or ground_truth_polygon.area <= 0
    ):
        return 0.0
    intersection = prediction_polygon.intersection(ground_truth_polygon).area
    return float(np.clip(
        intersection / ground_truth_polygon.area,
        0.0,
        1.0,
    ))


def strict_matches(
    boxes,
    probabilities,
    ground_truth_boxes,
    ground_truth_labels,
    confidence_threshold,
    angle_threshold,
):
    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    if len(ground_truth_boxes) == 0:
        return []
    distances = np.linalg.norm(
        boxes[:, :2, None] - ground_truth_boxes[:, :2].T,
        axis=1,
    )
    matches = []

    for class_id in ACTIVE_CLASSES:
        prediction_indices = np.where(
            (predicted_labels == class_id)
            & (confidences > confidence_threshold)
        )[0]
        ground_truth_indices = np.where(
            ground_truth_labels == class_id
        )[0]
        if len(prediction_indices) == 0 or len(ground_truth_indices) == 0:
            continue

        ordered_predictions = sorted(
            prediction_indices.tolist(),
            key=lambda index: float(confidences[index]),
            reverse=True,
        )
        used_ground_truth = set()
        threshold = DISTANCE_THRESHOLDS[class_id]

        for prediction_index in ordered_predictions:
            eligible = []
            for ground_truth_index in ground_truth_indices:
                ground_truth_index = int(ground_truth_index)
                if ground_truth_index in used_ground_truth:
                    continue
                center_error = float(
                    distances[prediction_index, ground_truth_index]
                )
                if center_error >= threshold:
                    continue
                angle_error = angle_difference(
                    float(boxes[prediction_index, 6]),
                    float(ground_truth_boxes[ground_truth_index, 6]),
                )
                if (
                    class_id in DIRECTIONAL_CLASSES
                    and angle_error > angle_threshold
                ):
                    continue
                eligible.append((
                    center_error,
                    ground_truth_index,
                    angle_error,
                ))

            if eligible:
                center_error, ground_truth_index, angle_error = min(eligible)
                used_ground_truth.add(ground_truth_index)
                matches.append({
                    "class_id": class_id,
                    "prediction_index": prediction_index,
                    "ground_truth_index": ground_truth_index,
                    "center_error": center_error,
                    "angle_error": angle_error,
                })
    return matches


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def safe_median(values):
    return float(np.median(values)) if values else float("nan")


def configure_axis(axis, title, labels):
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=25)
    axis.grid(axis="y", linestyle="--", alpha=0.3)


def annotate_counts(axis, values, y, color="#444444"):
    for index, value in enumerate(values):
        axis.text(
            index,
            y,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )


def save_recall_figure(stats, labels, out_dir, safe_tag):
    figure, axes = plt.subplots(4, 2, figsize=(18, 17), dpi=200)
    axes = axes.flatten()
    positions = np.arange(len(labels))

    for plot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[plot_index]
        total = np.array([
            stats[class_id][index]["gt_count"]
            for index in range(len(labels))
        ])
        matched = np.array([
            stats[class_id][index]["matched_count"]
            for index in range(len(labels))
        ])
        recall = np.divide(
            matched,
            total,
            out=np.full_like(matched, np.nan, dtype=np.float64),
            where=total > 0,
        )
        axis.bar(
            positions,
            recall,
            width=0.62,
            color=COLORS["positive"],
            alpha=0.85,
        )
        axis.plot(
            positions,
            recall,
            color="#1F4E79",
            marker="o",
            linewidth=2,
        )
        configure_axis(
            axis,
            f"{CLASS_NAMES[class_id]}: recall by GT-to-ego distance",
            labels,
        )
        axis.set_ylim(0.0, 1.08)
        axis.set_ylabel("Strict recall")
        for index, (match_count, gt_count) in enumerate(zip(matched, total)):
            if gt_count > 0:
                axis.text(
                    index,
                    min(float(recall[index]) + 0.035, 1.04),
                    f"{match_count}/{gt_count}",
                    ha="center",
                    fontsize=8,
                )
    axes[-1].axis("off")
    figure.suptitle(
        (
            f"Strict GT recall vs ego distance, {safe_tag}\n"
            "positive: confidence>0.2, same class, class-aware center "
            "threshold, angle<=45 deg"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    path = os.path.join(
        out_dir,
        f"gt_recall_vs_ego_distance_7cls_{safe_tag}.png",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def save_confidence_figure(stats, labels, out_dir, safe_tag):
    figure, axes = plt.subplots(4, 2, figsize=(18, 17), dpi=200)
    axes = axes.flatten()

    for plot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[plot_index]
        values = [
            stats[class_id][index]["confidence"]
            for index in range(len(labels))
        ]
        valid_positions = [
            index for index, item in enumerate(values) if len(item) > 0
        ]
        if valid_positions:
            boxplot = axis.boxplot(
                [values[index] for index in valid_positions],
                positions=valid_positions,
                widths=0.58,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#F28E2B",
                    "markeredgecolor": "#F28E2B",
                    "markersize": 4,
                },
            )
            for patch in boxplot["boxes"]:
                patch.set_facecolor(COLORS["positive"])
                patch.set_alpha(0.65)
        configure_axis(
            axis,
            f"{CLASS_NAMES[class_id]}: matched confidence",
            labels,
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Prediction confidence")
        counts = [len(item) for item in values]
        annotate_counts(axis, counts, 0.01)
        axis.text(
            0.99,
            0.03,
            "numbers: matched samples",
            transform=axis.transAxes,
            ha="right",
            fontsize=8,
            color="#555555",
        )
    axes[-1].axis("off")
    figure.suptitle(
        f"Matched prediction confidence vs GT-to-ego distance, {safe_tag}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(
        out_dir,
        f"matched_confidence_vs_ego_distance_7cls_{safe_tag}.png",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def save_geometry_figure(stats, labels, out_dir, safe_tag):
    figure, axes = plt.subplots(4, 2, figsize=(18, 17), dpi=200)
    axes = axes.flatten()
    positions = np.arange(len(labels))

    for plot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[plot_index]
        overlap = np.array([
            safe_mean(stats[class_id][index]["overlap_gt"])
            for index in range(len(labels))
        ])
        error = np.array([
            safe_mean(stats[class_id][index]["center_error"])
            for index in range(len(labels))
        ])
        counts = np.array([
            len(stats[class_id][index]["center_error"])
            for index in range(len(labels))
        ])
        axis.bar(
            positions,
            overlap,
            width=0.62,
            color=COLORS["overlap"],
            alpha=0.72,
            label="BEV intersection / GT area",
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Mean overlap ratio", color=COLORS["overlap"])
        axis.tick_params(axis="y", labelcolor=COLORS["overlap"])
        configure_axis(
            axis,
            f"{CLASS_NAMES[class_id]}: localization quality",
            labels,
        )
        annotate_counts(axis, counts, 0.01)

        error_axis = axis.twinx()
        error_axis.plot(
            positions,
            error,
            color=COLORS["error"],
            marker="o",
            linewidth=2,
            label="Mean center error",
        )
        threshold = DISTANCE_THRESHOLDS[class_id]
        error_axis.set_ylim(0.0, max(threshold * 1.35, 0.5))
        error_axis.set_ylabel(
            "Mean center error (m)",
            color=COLORS["error"],
        )
        error_axis.tick_params(axis="y", labelcolor=COLORS["error"])
        lines, line_labels = axis.get_legend_handles_labels()
        lines2, line_labels2 = error_axis.get_legend_handles_labels()
        axis.legend(
            lines + lines2,
            line_labels + line_labels2,
            loc="upper left",
            fontsize=8,
            frameon=False,
        )
    axes[-1].axis("off")
    figure.suptitle(
        (
            f"Localization quality vs GT-to-ego distance, {safe_tag}\n"
            "numbers at bottom: strict matched samples"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    path = os.path.join(
        out_dir,
        f"overlap_center_error_vs_ego_distance_7cls_{safe_tag}.png",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def save_similarity_figure(stats, labels, out_dir, safe_tag):
    figure, axes = plt.subplots(4, 2, figsize=(18, 17), dpi=200)
    axes = axes.flatten()
    positions = np.arange(len(labels))

    for plot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[plot_index]
        positive = np.array([
            safe_mean(stats[class_id][index]["positive_similarity"])
            for index in range(len(labels))
        ])
        negative = np.array([
            safe_mean(stats[class_id][index]["negative_similarity"])
            for index in range(len(labels))
        ])
        positive_counts = [
            len(stats[class_id][index]["positive_similarity"])
            for index in range(len(labels))
        ]
        negative_counts = [
            len(stats[class_id][index]["negative_similarity"])
            for index in range(len(labels))
        ]
        axis.plot(
            positions,
            positive,
            color=COLORS["positive"],
            marker="o",
            linewidth=2.2,
            label="Strict positives (GT distance)",
        )
        axis.plot(
            positions,
            negative,
            color=COLORS["negative"],
            marker="s",
            linewidth=2.2,
            label="Clear negatives (prediction distance)",
        )
        axis.axhline(
            0.7,
            color="#333333",
            linestyle="--",
            linewidth=1.3,
            label="similarity=0.7",
        )
        configure_axis(
            axis,
            f"{CLASS_NAMES[class_id]}: prototype similarity",
            labels,
        )
        axis.set_ylim(-0.05, 1.03)
        axis.set_ylabel("Mean maximum same-class similarity")
        axis.legend(loc="lower left", fontsize=7.5, frameon=False)
        for index, (positive_count, negative_count) in enumerate(
            zip(positive_counts, negative_counts)
        ):
            axis.text(
                index,
                0.01,
                f"P{positive_count}/N{negative_count}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#444444",
            )
    axes[-1].axis("off")
    figure.suptitle(
        (
            f"Positive and clear-negative prototype similarity vs ego "
            f"distance, {safe_tag}\n"
            "positive bins use GT distance; negative bins use predicted-box "
            "distance"
        ),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(
        out_dir,
        f"positive_negative_similarity_vs_ego_distance_7cls_{safe_tag}.png",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    safe_tag = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in args.tag
    ).strip("_") or "checkpoint"
    edges = parse_edges(args.distance_edges)
    labels = bin_labels(edges)
    prototypes = load_prototypes(args.proto_file)

    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    angle_threshold = math.radians(args.positive_angle_threshold_deg)

    stats = {
        class_id: {
            bin_index: {
                "gt_count": 0,
                "matched_count": 0,
                "confidence": [],
                "center_error": [],
                "angle_error_deg": [],
                "overlap_gt": [],
                "positive_similarity": [],
                "negative_similarity": [],
                "negative_confidence": [],
            }
            for bin_index in range(len(labels))
        }
        for class_id in ACTIVE_CLASSES
    }

    feature_files = sorted(
        [
            name
            for name in os.listdir(args.feature_dir)
            if name.startswith("sample_") and name.endswith("_features.npy")
        ],
        key=lambda name: int(name.split("_")[1]),
    )
    if len(feature_files) != len(infos):
        raise RuntimeError(
            f"feature samples ({len(feature_files)}) != infos ({len(infos)})"
        )

    for order, feature_name in enumerate(feature_files):
        sample_index = int(feature_name.split("_")[1])
        prefix = os.path.join(
            args.feature_dir,
            f"sample_{sample_index}",
        )
        features = np.load(f"{prefix}_features.npy").astype(np.float32)
        boxes = np.load(f"{prefix}_boxes.npy").astype(np.float32)
        scores = np.load(f"{prefix}_scores.npy").astype(np.float32)
        probabilities = sigmoid(scores)
        predicted_labels = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        normalized_features = normalize(features)
        matched_labels = np.load(os.path.join(
            args.feature_dir,
            f"th_{args.clear_negative_distance_threshold}",
            f"sample_{sample_index}_gt_labels.npy",
        )).astype(np.int64)

        info = infos[order]
        ground_truth_names = np.asarray(info["gt_names"])
        valid = np.array([
            str(name) in name_to_index
            for name in ground_truth_names
        ])
        if "valid_flag" in info:
            valid &= np.asarray(info["valid_flag"]).astype(bool)
        ground_truth_boxes = np.asarray(
            info["gt_boxes"],
            dtype=np.float32,
        )[valid]
        ground_truth_labels = np.array([
            name_to_index[str(name)]
            for name in ground_truth_names[valid]
        ], dtype=np.int64)
        ground_truth_distances = np.linalg.norm(
            ground_truth_boxes[:, :2],
            axis=1,
        ) if len(ground_truth_boxes) else np.empty(0, dtype=np.float32)

        for ground_truth_index, class_id in enumerate(ground_truth_labels):
            class_id = int(class_id)
            if class_id not in ACTIVE_CLASSES:
                continue
            bin_index = distance_bin(
                float(ground_truth_distances[ground_truth_index]),
                edges,
            )
            stats[class_id][bin_index]["gt_count"] += 1

        matches = strict_matches(
            boxes,
            probabilities,
            ground_truth_boxes,
            ground_truth_labels,
            args.positive_confidence_threshold,
            angle_threshold,
        )
        for match in matches:
            class_id = match["class_id"]
            prediction_index = match["prediction_index"]
            ground_truth_index = match["ground_truth_index"]
            bin_index = distance_bin(
                float(ground_truth_distances[ground_truth_index]),
                edges,
            )
            bucket = stats[class_id][bin_index]
            bucket["matched_count"] += 1
            bucket["confidence"].append(
                float(confidences[prediction_index])
            )
            bucket["center_error"].append(match["center_error"])
            bucket["angle_error_deg"].append(
                math.degrees(match["angle_error"])
            )
            bucket["overlap_gt"].append(overlap_over_gt_area(
                boxes[prediction_index],
                ground_truth_boxes[ground_truth_index],
            ))
            similarity = (
                normalized_features[prediction_index]
                @ prototypes[class_id].T
            ).max()
            bucket["positive_similarity"].append(float(similarity))

        if len(ground_truth_boxes):
            nearest_ground_truth_distance = np.linalg.norm(
                boxes[:, :2, None] - ground_truth_boxes[:, :2].T,
                axis=1,
            ).min(axis=1)
        else:
            nearest_ground_truth_distance = np.full(
                len(boxes),
                np.inf,
                dtype=np.float32,
            )
        clear_negative = (
            (matched_labels < 0)
            & (
                nearest_ground_truth_distance
                > args.clear_negative_distance_threshold
            )
            & np.isin(predicted_labels, ACTIVE_CLASSES)
        )
        prediction_ego_distances = np.linalg.norm(boxes[:, :2], axis=1)
        for prediction_index in np.where(clear_negative)[0]:
            class_id = int(predicted_labels[prediction_index])
            bin_index = distance_bin(
                float(prediction_ego_distances[prediction_index]),
                edges,
            )
            similarity = (
                normalized_features[prediction_index]
                @ prototypes[class_id].T
            ).max()
            stats[class_id][bin_index]["negative_similarity"].append(
                float(similarity)
            )
            stats[class_id][bin_index]["negative_confidence"].append(
                float(confidences[prediction_index])
            )

    summary_path = os.path.join(
        args.out_dir,
        f"ego_distance_metrics_7cls_{safe_tag}.csv",
    )
    fieldnames = [
        "class",
        "distance_bin_m",
        "gt_count",
        "matched_count",
        "recall",
        "confidence_mean",
        "confidence_median",
        "center_error_mean_m",
        "center_error_median_m",
        "angle_error_mean_deg",
        "overlap_over_gt_mean",
        "positive_similarity_count",
        "positive_similarity_mean",
        "positive_similarity_median",
        "clear_negative_count",
        "clear_negative_similarity_mean",
        "clear_negative_similarity_median",
        "clear_negative_confidence_mean",
    ]
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in ACTIVE_CLASSES:
            for bin_index, label in enumerate(labels):
                bucket = stats[class_id][bin_index]
                gt_count = bucket["gt_count"]
                matched_count = bucket["matched_count"]
                writer.writerow({
                    "class": CLASS_NAMES[class_id],
                    "distance_bin_m": label,
                    "gt_count": gt_count,
                    "matched_count": matched_count,
                    "recall": (
                        matched_count / gt_count
                        if gt_count
                        else float("nan")
                    ),
                    "confidence_mean": safe_mean(bucket["confidence"]),
                    "confidence_median": safe_median(bucket["confidence"]),
                    "center_error_mean_m": safe_mean(
                        bucket["center_error"]
                    ),
                    "center_error_median_m": safe_median(
                        bucket["center_error"]
                    ),
                    "angle_error_mean_deg": safe_mean(
                        bucket["angle_error_deg"]
                    ),
                    "overlap_over_gt_mean": safe_mean(
                        bucket["overlap_gt"]
                    ),
                    "positive_similarity_count": len(
                        bucket["positive_similarity"]
                    ),
                    "positive_similarity_mean": safe_mean(
                        bucket["positive_similarity"]
                    ),
                    "positive_similarity_median": safe_median(
                        bucket["positive_similarity"]
                    ),
                    "clear_negative_count": len(
                        bucket["negative_similarity"]
                    ),
                    "clear_negative_similarity_mean": safe_mean(
                        bucket["negative_similarity"]
                    ),
                    "clear_negative_similarity_median": safe_median(
                        bucket["negative_similarity"]
                    ),
                    "clear_negative_confidence_mean": safe_mean(
                        bucket["negative_confidence"]
                    ),
                })

    paths = [
        save_recall_figure(stats, labels, args.out_dir, safe_tag),
        save_confidence_figure(stats, labels, args.out_dir, safe_tag),
        save_geometry_figure(stats, labels, args.out_dir, safe_tag),
        save_similarity_figure(stats, labels, args.out_dir, safe_tag),
    ]
    print(f"SUMMARY={summary_path}")
    for path in paths:
        print(f"IMAGE={path}")
    for class_id in ACTIVE_CLASSES:
        total_gt = sum(
            stats[class_id][index]["gt_count"]
            for index in range(len(labels))
        )
        total_matched = sum(
            stats[class_id][index]["matched_count"]
            for index in range(len(labels))
        )
        print(
            f"{CLASS_NAMES[class_id]}: GT={total_gt}, "
            f"matched={total_matched}, "
            f"recall={total_matched / total_gt if total_gt else float('nan'):.4f}"
        )


if __name__ == "__main__":
    main()
