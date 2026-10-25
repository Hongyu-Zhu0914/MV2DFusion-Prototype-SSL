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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--angle-threshold-deg", type=float, default=45.0)
    return parser.parse_args()


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def angle_difference(first, second):
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


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
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return corners @ rotation.T + np.array([center_x, center_y], dtype=np.float32)


def overlap_over_gt(prediction, ground_truth):
    prediction_polygon = Polygon(get_bev_corners(prediction))
    ground_truth_polygon = Polygon(get_bev_corners(ground_truth))
    if not prediction_polygon.is_valid or not ground_truth_polygon.is_valid:
        return 0.0
    ground_truth_area = ground_truth_polygon.area
    if ground_truth_area <= 0:
        return 0.0
    intersection = prediction_polygon.intersection(ground_truth_polygon).area
    return float(np.clip(intersection / ground_truth_area, 0.0, 1.0))


def strict_greedy_match(
    boxes,
    scores,
    ground_truth_boxes,
    ground_truth_labels,
    confidence_threshold,
    angle_threshold,
):
    probabilities = sigmoid(scores)
    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
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
        ground_truth_indices = np.where(ground_truth_labels == class_id)[0]
        if len(prediction_indices) == 0 or len(ground_truth_indices) == 0:
            continue

        ordered_predictions = sorted(
            prediction_indices.tolist(),
            key=lambda index: float(confidences[index]),
            reverse=True,
        )
        used_ground_truth = set()
        distance_threshold = DISTANCE_THRESHOLDS[class_id]

        for prediction_index in ordered_predictions:
            eligible = []
            for ground_truth_index in ground_truth_indices:
                ground_truth_index = int(ground_truth_index)
                if ground_truth_index in used_ground_truth:
                    continue
                center_distance = float(
                    distances[prediction_index, ground_truth_index]
                )
                if center_distance >= distance_threshold:
                    continue
                if class_id in DIRECTIONAL_CLASSES:
                    difference = angle_difference(
                        float(boxes[prediction_index, 6]),
                        float(ground_truth_boxes[ground_truth_index, 6]),
                    )
                    if difference > angle_threshold:
                        continue
                eligible.append((center_distance, ground_truth_index))

            if not eligible:
                continue
            center_distance, ground_truth_index = min(eligible)
            used_ground_truth.add(ground_truth_index)
            matches.append(
                (
                    prediction_index,
                    ground_truth_index,
                    class_id,
                    float(confidences[prediction_index]),
                    center_distance,
                )
            )
    return matches


def draw_histograms(
    values_by_class,
    bins,
    output_path,
    figure_title,
    x_label,
    show_thresholds=False,
):
    figure, axes = plt.subplots(4, 2, figsize=(20, 17), dpi=200)
    axes = axes.flatten()
    centers = (bins[:-1] + bins[1:]) / 2
    width = np.diff(bins) * 0.86

    for subplot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[subplot_index]
        values = np.asarray(values_by_class[class_id], dtype=np.float32)
        counts, _ = np.histogram(values, bins=bins)
        axis.bar(
            centers,
            counts,
            width=width,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
        )
        if show_thresholds:
            threshold = DISTANCE_THRESHOLDS[class_id]
            axis.axvline(
                threshold,
                color="#D62728",
                linestyle="--",
                linewidth=2,
                label=f"threshold={threshold:.2f} m",
            )
            axis.legend(loc="upper right", frameon=False, fontsize=9)
        axis.set_title(
            f"{CLASS_NAMES[class_id]}  (n={len(values)})",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Positive sample count")
        axis.set_xlim(float(bins[0]), float(bins[-1]))
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].axis("off")
    figure.suptitle(figure_title, fontsize=18, fontweight="bold", y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.975])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    overlaps = defaultdict(list)
    distances = defaultdict(list)
    confidences = defaultdict(list)
    feature_files = sorted(
        [
            name
            for name in os.listdir(args.feature_dir)
            if name.startswith("sample_") and name.endswith("_features.npy")
        ],
        key=lambda name: int(name.split("_")[1]),
    )
    angle_threshold = math.radians(args.angle_threshold_deg)

    for order, feature_name in enumerate(feature_files):
        sample_index = int(feature_name.split("_")[1])
        boxes = np.load(
            os.path.join(args.feature_dir, f"sample_{sample_index}_boxes.npy")
        ).astype(np.float32)
        scores = np.load(
            os.path.join(args.feature_dir, f"sample_{sample_index}_scores.npy")
        ).astype(np.float32)
        info = infos[order]
        ground_truth_names = np.asarray(info["gt_names"])
        valid = np.array(
            [str(name) in name_to_index for name in ground_truth_names]
        )
        ground_truth_boxes = np.asarray(
            info["gt_boxes"],
            dtype=np.float32,
        )[valid]
        ground_truth_labels = np.array(
            [name_to_index[str(name)] for name in ground_truth_names[valid]],
            dtype=np.int64,
        )
        if len(ground_truth_boxes) == 0:
            continue

        matches = strict_greedy_match(
            boxes,
            scores,
            ground_truth_boxes,
            ground_truth_labels,
            args.confidence_threshold,
            angle_threshold,
        )
        for prediction_index, ground_truth_index, class_id, confidence, distance in matches:
            overlaps[class_id].append(
                overlap_over_gt(
                    boxes[prediction_index],
                    ground_truth_boxes[ground_truth_index],
                )
            )
            distances[class_id].append(distance)
            confidences[class_id].append(confidence)

    overlap_bins = np.linspace(0.0, 1.0, 11)
    distance_bins = np.linspace(0.0, 1.5, 16)
    confidence_bins = np.linspace(0.2, 1.0, 17)
    overlap_path = os.path.join(
        args.out_dir,
        "positive_overlap_over_gt_hist_7cls_iter5000.png",
    )
    distance_path = os.path.join(
        args.out_dir,
        "positive_center_distance_hist_7cls_iter5000.png",
    )
    confidence_path = os.path.join(
        args.out_dir,
        "positive_confidence_hist_7cls_iter5000.png",
    )
    draw_histograms(
        overlaps,
        overlap_bins,
        overlap_path,
        (
            "Strict positive samples: BEV intersection / GT area\n"
            "confidence>0.2, class-aware one-to-one greedy matching, "
            "class-specific distance thresholds, angle difference<=45 deg"
        ),
        "BEV overlap ratio (intersection / GT area)",
    )
    draw_histograms(
        distances,
        distance_bins,
        distance_path,
        (
            "Strict positive samples: center-distance distribution\n"
            "confidence>0.2, class-aware one-to-one greedy matching, "
            "class-specific distance thresholds, angle difference<=45 deg"
        ),
        "BEV center distance (m)",
        show_thresholds=True,
    )
    draw_histograms(
        confidences,
        confidence_bins,
        confidence_path,
        (
            "Strict positive samples: confidence distribution\n"
            "confidence=max sigmoid(class logits), class-aware one-to-one "
            "greedy matching, class-specific distance thresholds, "
            "angle difference<=45 deg"
        ),
        "Prediction confidence",
    )

    summary_path = os.path.join(
        args.out_dir,
        "strict_positive_geometry_summary_iter5000.csv",
    )
    with open(summary_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class",
                "positive_count",
                "distance_threshold_m",
                "overlap_mean",
                "overlap_median",
                "distance_mean_m",
                "distance_median_m",
                "confidence_mean",
            ]
        )
        for class_id in ACTIVE_CLASSES:
            class_overlaps = np.asarray(overlaps[class_id], dtype=np.float32)
            class_distances = np.asarray(distances[class_id], dtype=np.float32)
            class_confidences = np.asarray(confidences[class_id], dtype=np.float32)
            writer.writerow(
                [
                    CLASS_NAMES[class_id],
                    len(class_overlaps),
                    DISTANCE_THRESHOLDS[class_id],
                    float(class_overlaps.mean()) if len(class_overlaps) else np.nan,
                    float(np.median(class_overlaps)) if len(class_overlaps) else np.nan,
                    float(class_distances.mean()) if len(class_distances) else np.nan,
                    float(np.median(class_distances)) if len(class_distances) else np.nan,
                    float(class_confidences.mean()) if len(class_confidences) else np.nan,
                ]
            )

    print(f"OVERLAP_IMAGE={overlap_path}")
    print(f"DISTANCE_IMAGE={distance_path}")
    print(f"CONFIDENCE_IMAGE={confidence_path}")
    print(f"SUMMARY={summary_path}")
    for class_id in ACTIVE_CLASSES:
        print(
            f"{CLASS_NAMES[class_id]}: n={len(overlaps[class_id])}, "
            f"overlap_mean={np.mean(overlaps[class_id]):.4f}, "
            f"distance_mean={np.mean(distances[class_id]):.4f}"
        )


if __name__ == "__main__":
    main()
