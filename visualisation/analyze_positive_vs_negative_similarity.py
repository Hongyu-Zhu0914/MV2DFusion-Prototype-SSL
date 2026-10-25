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
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict-positive and clear-negative maximum similarities "
            "to the 10 prototypes of each predicted class."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--proto-file", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tag",
        default="checkpoint",
        help="Checkpoint label used in the figure title and output filenames.",
    )
    parser.add_argument("--positive-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--positive-angle-threshold-deg", type=float, default=45.0)
    parser.add_argument("--clear-negative-distance-threshold", type=float, default=2.0)
    parser.add_argument("--similarity-reference", type=float, default=0.7)
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
    os.makedirs(args.out_dir, exist_ok=True)
    safe_tag = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in args.tag
    ).strip("_") or "checkpoint"
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
        normalized_features = normalize(features)

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
    figure, axes = plt.subplots(4, 2, figsize=(20, 17), dpi=200)
    axes = axes.flatten()
    summary_rows = []

    for subplot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[subplot_index]
        positives = np.asarray(
            positive_similarity[class_id],
            dtype=np.float32,
        )
        negatives = np.asarray(
            negative_similarity[class_id],
            dtype=np.float32,
        )
        positive_counts, _ = np.histogram(positives, bins=bins)
        negative_counts, _ = np.histogram(negatives, bins=bins)
        axis.bar(
            centers - bar_width / 2,
            positive_counts,
            width=bar_width,
            color="#4C78A8",
            label="Strict positives",
        )
        axis.bar(
            centers + bar_width / 2,
            negative_counts,
            width=bar_width,
            color="#E45756",
            label="Clear negatives",
        )
        axis.axvline(
            args.similarity_reference,
            color="#1F1F1F",
            linestyle="--",
            linewidth=2,
            label=f"reference={args.similarity_reference:.2f}",
        )
        axis.set_title(
            (
                f"{CLASS_NAMES[class_id]}\n"
                f"positive: n={len(positives)}, mean={positives.mean():.3f}; "
                f"negative: n={len(negatives)}, mean={negatives.mean():.3f}"
            ),
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlabel(
            "Maximum similarity to 10 prototypes of predicted class"
        )
        axis.set_ylabel("Sample count")
        axis.set_xlim(-1.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left", frameon=False, fontsize=8)
        summary_rows.append(
            {
                "class": CLASS_NAMES[class_id],
                "positive_count": len(positives),
                "positive_mean_similarity": float(positives.mean()),
                "positive_median_similarity": float(np.median(positives)),
                "positive_ratio_gt_0_7": float(
                    (positives > args.similarity_reference).mean()
                ),
                "clear_negative_count": len(negatives),
                "clear_negative_mean_similarity": float(negatives.mean()),
                "clear_negative_median_similarity": float(
                    np.median(negatives)
                ),
                "clear_negative_ratio_gt_0_7": float(
                    (negatives > args.similarity_reference).mean()
                ),
                "positive_minus_negative_mean": float(
                    positives.mean() - negatives.mean()
                ),
            }
        )

    axes[-1].axis("off")
    figure.suptitle(
        (
            f"Strict-positive vs clear-negative prototype similarity, {args.tag}\n"
            "positive: confidence>0.2 + class-aware thresholds + angle<=45 deg; "
            "negative: unmatched + nearest GT distance>2 m, no confidence threshold"
        ),
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    image_path = os.path.join(
        args.out_dir,
        f"positive_vs_clear_negative_similarity_hist_7cls_{safe_tag}.png",
    )
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    summary_path = os.path.join(
        args.out_dir,
        f"positive_vs_clear_negative_similarity_summary_{safe_tag}.csv",
    )
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"IMAGE={image_path}")
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
