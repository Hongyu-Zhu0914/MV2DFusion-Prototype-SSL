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
ACTIVE_CLASSES = list(range(10))
DISTANCE_THRESHOLDS = {
    0: 1.0,
    1: 1.2,
    2: 1.5,
    3: 1.5,
    4: 1.5,
    5: 0.5,
    6: 0.5,
    7: 0.4,
    8: 0.35,
    9: 0.25,
}
DIRECTIONAL_CLASSES = {0, 1, 2, 3, 4, 6, 7}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--proto-file", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--negative-confidence-lower", type=float, default=None)
    parser.add_argument("--confidence-upper", type=float, default=None)
    parser.add_argument("--positive-angle-threshold-deg", type=float, default=45.0)
    parser.add_argument("--clear-negative-distance-threshold", type=float, default=2.0)
    parser.add_argument("--similarity-reference", type=float, default=0.9,
                        help="Legacy reference for classes other than bicycle; not re-calibrated here")
    parser.add_argument("--bicycle-reference", type=float, default=0.94)
    return parser.parse_args()


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def normalize(values):
    return values / (
        np.linalg.norm(values, axis=-1, keepdims=True) + 1e-8
    )


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


def strict_positive_indices(
    boxes,
    probabilities,
    ground_truth_boxes,
    ground_truth_labels,
    confidence_threshold,
    angle_threshold,
    confidence_upper=None,
):
    predicted_labels = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    distances = np.linalg.norm(
        boxes[:, :2, None] - ground_truth_boxes[:, :2].T,
        axis=1,
    )
    positive_by_class = defaultdict(list)

    for class_id in ACTIVE_CLASSES:
        prediction_indices = np.where(
            (predicted_labels == class_id)
            & (confidences > confidence_threshold)
            & ((confidences < confidence_upper) if confidence_upper is not None else True)
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

            if eligible:
                _, ground_truth_index = min(eligible)
                used_ground_truth.add(ground_truth_index)
                positive_by_class[class_id].append(prediction_index)
    return positive_by_class


def append_similarity(
    output,
    normalized_features,
    predicted_labels,
    selected_indices,
    prototypes,
):
    for class_id in ACTIVE_CLASSES:
        class_indices = np.asarray(
            selected_indices.get(class_id, []),
            dtype=np.int64,
        )
        if len(class_indices) == 0:
            continue
        class_similarity = (
            normalized_features[class_indices] @ prototypes[class_id].T
        ).max(axis=1)
        output[class_id].extend(class_similarity.tolist())


def main():
    args = parse_args()
    if args.confidence_upper is not None:
        if not 0 < args.positive_confidence_threshold < args.confidence_upper <= 1:
            raise ValueError("Require 0 < positive-confidence-threshold < confidence-upper <= 1")
        if args.negative_confidence_lower is not None and not 0 <= args.negative_confidence_lower < args.confidence_upper:
            raise ValueError("Require 0 <= negative-confidence-lower < confidence-upper")
    os.makedirs(args.out_dir, exist_ok=True)
    prototypes = load_prototypes(args.proto_file)
    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    positive_similarity = defaultdict(list)
    negative_similarity = defaultdict(list)
    feature_files = sorted(
        [
            name
            for name in os.listdir(args.feature_dir)
            if name.startswith("sample_") and name.endswith("_features.npy")
        ],
        key=lambda name: int(name.split("_")[1]),
    )
    actual_indices = [int(name.split("_")[1]) for name in feature_files]
    if actual_indices != list(range(len(infos))):
        raise RuntimeError(
            f"Feature indices must exactly cover {len(infos)} PKL frames; "
            f"found {len(actual_indices)}"
        )
    num_frames = len(infos)
    angle_threshold = math.radians(
        args.positive_angle_threshold_deg
    )

    for order, feature_name in enumerate(feature_files):
        sample_index = int(feature_name.split("_")[1])
        features = np.load(
            os.path.join(
                args.feature_dir,
                f"sample_{sample_index}_features.npy",
            )
        ).astype(np.float32)
        boxes = np.load(
            os.path.join(
                args.feature_dir,
                f"sample_{sample_index}_boxes.npy",
            )
        ).astype(np.float32)
        scores = np.load(
            os.path.join(
                args.feature_dir,
                f"sample_{sample_index}_scores.npy",
            )
        ).astype(np.float32)
        matched_labels = np.load(
            os.path.join(
                args.feature_dir,
                f"th_{args.clear_negative_distance_threshold}",
                f"sample_{sample_index}_gt_labels.npy",
            )
        ).astype(np.int64)
        probabilities = sigmoid(scores)
        predicted_labels = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        normalized_features = normalize(features)

        info = infos[order]
        ground_truth_names = np.asarray(info["gt_names"])
        valid = np.array(
            [str(name) in name_to_index for name in ground_truth_names],
            dtype=bool,
        )
        ground_truth_boxes = np.asarray(
            info["gt_boxes"],
            dtype=np.float32,
        )[valid]
        ground_truth_labels = np.array(
            [
                name_to_index[str(name)]
                for name in ground_truth_names[valid]
            ],
            dtype=np.int64,
        )

        if len(ground_truth_boxes) > 0:
            positive_by_class = strict_positive_indices(
                boxes,
                probabilities,
                ground_truth_boxes,
                ground_truth_labels,
                args.positive_confidence_threshold,
                angle_threshold,
                args.confidence_upper,
            )
            append_similarity(
                positive_similarity,
                normalized_features,
                predicted_labels,
                positive_by_class,
                prototypes,
            )
            nearest_distance = np.linalg.norm(
                boxes[:, :2, None] - ground_truth_boxes[:, :2].T,
                axis=1,
            ).min(axis=1)
        else:
            nearest_distance = np.full(
                len(boxes),
                np.inf,
                dtype=np.float32,
            )

        clear_negative = (
            (matched_labels < 0)
            & (
                nearest_distance
                > args.clear_negative_distance_threshold
            )
            & np.isin(predicted_labels, ACTIVE_CLASSES)
            & ((confidences > args.negative_confidence_lower)
               if args.negative_confidence_lower is not None else True)
            & ((confidences < args.confidence_upper)
               if args.confidence_upper is not None else True)
        )
        for class_id in ACTIVE_CLASSES:
            class_indices = np.where(
                clear_negative & (predicted_labels == class_id)
            )[0]
            if len(class_indices) == 0:
                continue
            class_similarity = (
                normalized_features[class_indices]
                @ prototypes[class_id].T
            ).max(axis=1)
            negative_similarity[class_id].extend(
                class_similarity.tolist()
            )

    bins = np.linspace(-1.0, 1.0, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    bin_width = float(bins[1] - bins[0])
    bar_width = bin_width * 0.42
    figure, axes = plt.subplots(5, 2, figsize=(22, 21), dpi=180)
    axes = axes.flatten()
    summary_rows = []

    for subplot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[subplot_index]
        class_reference = (args.bicycle_reference if class_id == 7
                           else args.similarity_reference)
        positives = np.asarray(
            positive_similarity[class_id],
            dtype=np.float32,
        )
        negatives = np.asarray(
            negative_similarity[class_id],
            dtype=np.float32,
        )
        positive_mean = float(positives.mean()) if len(positives) else float("nan")
        negative_mean = float(negatives.mean()) if len(negatives) else float("nan")
        positive_median = float(np.median(positives)) if len(positives) else float("nan")
        negative_median = float(np.median(negatives)) if len(negatives) else float("nan")
        positive_above = float((positives > class_reference).mean()) if len(positives) else float("nan")
        negative_above = float((negatives > class_reference).mean()) if len(negatives) else float("nan")
        positive_counts, _ = np.histogram(positives, bins=bins)
        negative_counts, _ = np.histogram(negatives, bins=bins)
        axis.bar(
            centers - bar_width / 2,
            positive_counts / num_frames,
            width=bar_width,
            color="#4C78A8",
            label="Strict positives",
        )
        axis.bar(
            centers + bar_width / 2,
            negative_counts / num_frames,
            width=bar_width,
            color="#E45756",
            label="Clear negatives",
        )
        axis.axvline(
            class_reference,
            color="#1F1F1F",
            linestyle="--",
            linewidth=2,
            label=f"legacy reference={class_reference:.2f}",
        )
        axis.set_title(
            (
                f"{CLASS_NAMES[class_id]}\n"
                f"positive: n={len(positives)}, mean={positive_mean:.3f}; "
                f"negative: n={len(negatives)}, mean={negative_mean:.3f}"
            ),
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel(
            "Maximum similarity to 10 prototypes of predicted class"
        )
        axis.set_ylabel("Samples per frame / bin")
        axis.set_xlim(-1.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left", frameon=False, fontsize=8)
        summary_rows.append(
            {
                "class": CLASS_NAMES[class_id],
                "frames": num_frames,
                "legacy_similarity_reference": class_reference,
                "positive_count": len(positives),
                "positive_per_frame": len(positives) / num_frames,
                "positive_mean_similarity": positive_mean,
                "positive_median_similarity": positive_median,
                "positive_ratio_above_reference": positive_above,
                "clear_negative_count": len(negatives),
                "clear_negative_per_frame": len(negatives) / num_frames,
                "clear_negative_mean_similarity": negative_mean,
                "clear_negative_median_similarity": negative_median,
                "clear_negative_ratio_above_reference": negative_above,
                "positive_minus_negative_mean": positive_mean - negative_mean,
            }
        )

    positive_confidence_rule = f"confidence>{args.positive_confidence_threshold:g}"
    negative_confidence_rule = (
        f"confidence>{args.negative_confidence_lower:g}"
        if args.negative_confidence_lower is not None else "no confidence lower bound"
    )
    if args.confidence_upper is not None:
        positive_confidence_rule += f" and <{args.confidence_upper:g}"
        negative_confidence_rule += f" and <{args.confidence_upper:g}"
    figure.suptitle(
        (
            "Strict-positive vs clear-negative prototype similarity, iter_6000 (10 classes)\n"
            f"positive: {positive_confidence_rule} + class-aware thresholds + angle<=45 deg; "
            f"negative: {negative_confidence_rule} + unmatched + nearest GT distance>2 m"
        ),
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    output_suffix = (
        f"_conf_{args.positive_confidence_threshold:g}_to_{args.confidence_upper:g}"
        if args.confidence_upper is not None else ""
    )
    image_path = os.path.join(
        args.out_dir,
        f"positive_vs_clear_negative_similarity_hist_10cls_iter6000{output_suffix}.png",
    )
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    density_figure, density_axes = plt.subplots(5, 2, figsize=(22, 21), dpi=180)
    for class_id, density_axis in enumerate(density_axes.flatten()):
        positives = np.asarray(positive_similarity[class_id], dtype=np.float32)
        negatives = np.asarray(negative_similarity[class_id], dtype=np.float32)
        if len(positives):
            density_axis.hist(positives, bins=bins, density=True,
                              histtype="step", linewidth=2.2, color="#4C78A8",
                              label=f"Strict positives (n={len(positives)})")
        if len(negatives):
            density_axis.hist(negatives, bins=bins, density=True,
                              histtype="step", linewidth=2.2, color="#E45756",
                              label=f"Clear negatives (n={len(negatives)})")
        class_reference = (args.bicycle_reference if class_id == 7
                           else args.similarity_reference)
        density_axis.axvline(class_reference, color="#1F1F1F", linestyle="--",
                             linewidth=2, label=f"legacy reference={class_reference:.2f}")
        density_axis.set_title(CLASS_NAMES[class_id], fontsize=12, fontweight="bold")
        density_axis.set_xlabel("Maximum similarity to 10 prototypes of predicted class")
        density_axis.set_ylabel("Within-group probability density")
        density_axis.set_xlim(-1.0, 1.0)
        density_axis.grid(axis="y", linestyle="--", alpha=0.3)
        density_axis.legend(loc="upper left", frameon=False, fontsize=8)
    density_figure.suptitle(
        f"Within-group normalized similarity distribution, iter_6000 (10 classes)\n"
        f"positive: {positive_confidence_rule}; negative: {negative_confidence_rule}",
        fontsize=18, fontweight="bold", y=0.995,
    )
    density_figure.tight_layout(rect=[0, 0, 1, 0.97])
    density_path = os.path.join(
        args.out_dir,
        f"positive_vs_clear_negative_similarity_density_10cls_iter6000{output_suffix}.png",
    )
    density_figure.savefig(density_path, bbox_inches="tight")
    plt.close(density_figure)

    summary_path = os.path.join(
        args.out_dir,
        f"positive_vs_clear_negative_similarity_summary_10cls_iter6000{output_suffix}.csv",
    )
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"IMAGE={image_path}")
    print(f"DENSITY_IMAGE={density_path}")
    print(f"SUMMARY={summary_path}")
    for row in summary_rows:
        print(
            f"{row['class']}: positive={row['positive_count']} "
            f"({row['positive_mean_similarity']:.4f}), "
            f"negative={row['clear_negative_count']} "
            f"({row['clear_negative_mean_similarity']:.4f}), "
            f"gap={row['positive_minus_negative_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
