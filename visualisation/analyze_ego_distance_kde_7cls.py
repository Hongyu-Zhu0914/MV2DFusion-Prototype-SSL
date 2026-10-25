import argparse
import csv
import math
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_ego_distance_performance import (
    ACTIVE_CLASSES,
    CLASS_NAMES,
    load_prototypes,
    normalize,
    sigmoid,
    strict_matches,
)


CLASS_COLORS = {
    0: "#4C78A8",
    1: "#F28E2B",
    3: "#59A14F",
    6: "#E15759",
    7: "#B279A2",
    8: "#76B7B2",
    9: "#EDC948",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot GT-to-ego distance KDE curves and the mean distance of "
            "strict positive samples assigned to each prototype."
        )
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--proto-file", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="checkpoint")
    parser.add_argument(
        "--positive-confidence-threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--positive-angle-threshold-deg",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=70.0,
        help="Fixed x-axis maximum in meters for checkpoint comparison.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=600,
    )
    return parser.parse_args()


def safe_tag(value):
    result = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in value
    ).strip("_")
    return result or "checkpoint"


def gaussian_kde(values, grid):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(grid)
    if len(values) == 1:
        bandwidth = 1.0
    else:
        standard_deviation = float(np.std(values, ddof=1))
        interquartile_range = float(
            np.subtract(*np.percentile(values, [75, 25]))
        )
        robust_scale = min(
            standard_deviation,
            interquartile_range / 1.34
            if interquartile_range > 0
            else standard_deviation,
        )
        if not np.isfinite(robust_scale) or robust_scale <= 1e-6:
            robust_scale = max(standard_deviation, 1.0)
        bandwidth = max(
            0.9 * robust_scale * len(values) ** (-0.2),
            0.8,
        )
    normalized = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * normalized**2).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    return density


def load_data(args):
    prototypes = load_prototypes(args.proto_file)
    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    name_to_index = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }
    angle_threshold = math.radians(
        args.positive_angle_threshold_deg
    )
    class_data = {
        class_id: {
            "all_gt_distances": [],
            "positive_distances": [],
            "prototype_distances": [[] for _ in range(10)],
        }
        for class_id in ACTIVE_CLASSES
    }

    feature_files = sorted(
        [
            name
            for name in os.listdir(args.feature_dir)
            if name.startswith("sample_")
            and name.endswith("_features.npy")
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
        normalized_features = normalize(features)

        info = infos[order]
        ground_truth_names = np.asarray(info["gt_names"])
        valid = np.array(
            [
                str(name) in name_to_index
                for name in ground_truth_names
            ]
        )
        if "valid_flag" in info:
            valid &= np.asarray(info["valid_flag"]).astype(bool)
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
        ground_truth_distances = (
            np.linalg.norm(ground_truth_boxes[:, :2], axis=1)
            if len(ground_truth_boxes)
            else np.empty(0, dtype=np.float32)
        )

        for ground_truth_index, class_id in enumerate(
            ground_truth_labels
        ):
            class_id = int(class_id)
            if class_id in ACTIVE_CLASSES:
                class_data[class_id]["all_gt_distances"].append(
                    float(ground_truth_distances[ground_truth_index])
                )

        matches = strict_matches(
            boxes,
            probabilities,
            ground_truth_boxes,
            ground_truth_labels,
            args.positive_confidence_threshold,
            angle_threshold,
        )
        for match in matches:
            class_id = int(match["class_id"])
            prediction_index = int(match["prediction_index"])
            ground_truth_index = int(match["ground_truth_index"])
            distance = float(
                ground_truth_distances[ground_truth_index]
            )
            similarities = (
                normalized_features[prediction_index]
                @ prototypes[class_id].T
            )
            prototype_index = int(np.argmax(similarities))
            class_data[class_id]["positive_distances"].append(distance)
            class_data[class_id]["prototype_distances"][
                prototype_index
            ].append(distance)

    return class_data


def write_summary(class_data, path):
    fieldnames = [
        "class",
        "prototype",
        "assigned_positive_count",
        "mean_ego_distance_m",
        "median_ego_distance_m",
        "std_ego_distance_m",
        "minimum_ego_distance_m",
        "maximum_ego_distance_m",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in ACTIVE_CLASSES:
            for prototype_index, values in enumerate(
                class_data[class_id]["prototype_distances"]
            ):
                array = np.asarray(values, dtype=np.float64)
                writer.writerow(
                    {
                        "class": CLASS_NAMES[class_id],
                        "prototype": prototype_index,
                        "assigned_positive_count": len(array),
                        "mean_ego_distance_m": (
                            float(np.mean(array))
                            if len(array)
                            else float("nan")
                        ),
                        "median_ego_distance_m": (
                            float(np.median(array))
                            if len(array)
                            else float("nan")
                        ),
                        "std_ego_distance_m": (
                            float(np.std(array))
                            if len(array)
                            else float("nan")
                        ),
                        "minimum_ego_distance_m": (
                            float(np.min(array))
                            if len(array)
                            else float("nan")
                        ),
                        "maximum_ego_distance_m": (
                            float(np.max(array))
                            if len(array)
                            else float("nan")
                        ),
                    }
                )


def prototype_text(prototype_distances):
    entries = []
    for prototype_index, values in enumerate(prototype_distances):
        if values:
            entries.append(
                f"P{prototype_index}: {np.mean(values):.1f}m"
                f" (n={len(values)})"
            )
        else:
            entries.append(f"P{prototype_index}: -- (n=0)")
    return "\n".join(
        [
            "   ".join(entries[:5]),
            "   ".join(entries[5:]),
        ]
    )


def save_figure(class_data, args, output_path):
    grid = np.linspace(0.0, args.max_distance, args.grid_points)
    prototype_colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(24, 20),
        dpi=220,
        sharex=True,
    )
    axes = axes.flatten()

    for plot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[plot_index]
        all_distances = np.asarray(
            class_data[class_id]["all_gt_distances"],
            dtype=np.float64,
        )
        positive_distances = np.asarray(
            class_data[class_id]["positive_distances"],
            dtype=np.float64,
        )
        all_density = gaussian_kde(all_distances, grid)
        positive_density = gaussian_kde(positive_distances, grid)
        class_color = CLASS_COLORS[class_id]

        axis.plot(
            grid,
            all_density,
            color="#7F8C8D",
            linewidth=2.0,
            linestyle="--",
            label=f"All valid GT (n={len(all_distances)})",
        )
        axis.plot(
            grid,
            positive_density,
            color=class_color,
            linewidth=2.8,
            label=f"Strict positives (n={len(positive_distances)})",
        )
        axis.fill_between(
            grid,
            positive_density,
            color=class_color,
            alpha=0.12,
        )

        density_maximum = max(
            float(np.max(all_density)),
            float(np.max(positive_density)),
            1e-4,
        )
        prototype_distances = class_data[class_id][
            "prototype_distances"
        ]
        for prototype_index, values in enumerate(prototype_distances):
            if not values:
                continue
            mean_distance = float(np.mean(values))
            color = prototype_colors[prototype_index]
            axis.axvline(
                mean_distance,
                color=color,
                linewidth=1.4,
                alpha=0.9,
            )
            level = density_maximum * (
                0.94 - 0.075 * (prototype_index % 5)
            )
            horizontal_alignment = (
                "right" if prototype_index % 2 == 0 else "left"
            )
            axis.text(
                mean_distance,
                level,
                f"P{prototype_index} {mean_distance:.1f}m",
                rotation=90,
                rotation_mode="anchor",
                ha=horizontal_alignment,
                va="top",
                fontsize=6.6,
                color=color,
                clip_on=True,
            )

        axis.text(
            0.01,
            0.015,
            prototype_text(prototype_distances),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.4,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#DDDDDD",
                "alpha": 0.88,
            },
        )
        axis.set_title(
            CLASS_NAMES[class_id],
            fontsize=13,
            fontweight="bold",
        )
        axis.set_xlim(0.0, args.max_distance)
        axis.set_ylim(0.0, density_maximum * 1.08)
        axis.set_ylabel("Density")
        axis.grid(
            linestyle="--",
            linewidth=0.6,
            alpha=0.3,
        )
        axis.legend(
            loc="upper right",
            fontsize=8,
            frameon=False,
        )

    axes[-1].axis("off")
    for axis in axes[:-1]:
        axis.set_xlabel("GT center distance to ego vehicle (m)")
    figure.suptitle(
        (
            "Seven-class GT-to-ego distance distribution and "
            "10 prototype mean distances\n"
            f"{args.tag} | strict positive: score>{args.positive_confidence_threshold:g}, "
            "same class, class-aware center threshold, angle<=45 deg"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.997,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = safe_tag(args.tag)
    class_data = load_data(args)
    image_path = os.path.join(
        args.out_dir,
        f"positive_ego_distance_kde_with_10proto_means_7cls_{tag}.png",
    )
    summary_path = os.path.join(
        args.out_dir,
        f"positive_ego_distance_proto_means_7cls_{tag}.csv",
    )
    write_summary(class_data, summary_path)
    save_figure(class_data, args, image_path)

    print(f"IMAGE={image_path}")
    print(f"SUMMARY={summary_path}")
    for class_id in ACTIVE_CLASSES:
        used = sum(
            bool(values)
            for values in class_data[class_id]["prototype_distances"]
        )
        print(
            f"{CLASS_NAMES[class_id]}: "
            f"GT={len(class_data[class_id]['all_gt_distances'])}, "
            f"positive={len(class_data[class_id]['positive_distances'])}, "
            f"used_prototypes={used}/10"
        )


if __name__ == "__main__":
    main()

