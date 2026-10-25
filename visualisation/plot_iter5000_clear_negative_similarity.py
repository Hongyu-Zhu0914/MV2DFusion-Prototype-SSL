import argparse
import csv
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--proto-file", required=True)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--similarity-reference", type=float, default=0.7)
    return parser.parse_args()


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def normalize(values):
    return values / (
        np.linalg.norm(values, axis=-1, keepdims=True) + 1e-8
    )


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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prototypes = load_prototypes(args.proto_file)
    with open(args.pkl, "rb") as file:
        data = pickle.load(file)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    name_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    similarities_by_class = defaultdict(list)
    confidences_by_class = defaultdict(list)
    feature_files = sorted(
        [
            name
            for name in os.listdir(args.feature_dir)
            if name.startswith("sample_") and name.endswith("_features.npy")
        ],
        key=lambda name: int(name.split("_")[1]),
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
                f"th_{args.distance_threshold}",
                f"sample_{sample_index}_gt_labels.npy",
            )
        ).astype(np.int64)

        probabilities = sigmoid(scores)
        predicted_labels = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
        info = infos[order]
        ground_truth_names = np.asarray(info["gt_names"])
        valid = np.array(
            [str(name) in name_to_index for name in ground_truth_names]
        )
        ground_truth_boxes = np.asarray(
            info["gt_boxes"],
            dtype=np.float32,
        )[valid]

        if len(ground_truth_boxes) > 0:
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
            & (nearest_distance > args.distance_threshold)
            & np.isin(predicted_labels, ACTIVE_CLASSES)
        )
        if not np.any(clear_negative):
            continue

        normalized_features = normalize(features[clear_negative])
        negative_labels = predicted_labels[clear_negative]
        negative_confidences = confidences[clear_negative]
        for class_id in ACTIVE_CLASSES:
            class_mask = negative_labels == class_id
            if not np.any(class_mask):
                continue
            class_similarity = (
                normalized_features[class_mask] @ prototypes[class_id].T
            ).max(axis=1)
            similarities_by_class[class_id].extend(
                class_similarity.tolist()
            )
            confidences_by_class[class_id].extend(
                negative_confidences[class_mask].tolist()
            )

    bins = np.linspace(-1.0, 1.0, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins) * 0.86
    figure, axes = plt.subplots(4, 2, figsize=(20, 17), dpi=200)
    axes = axes.flatten()

    summary_rows = []
    all_similarities = []
    all_confidences = []
    for subplot_index, class_id in enumerate(ACTIVE_CLASSES):
        axis = axes[subplot_index]
        similarities = np.asarray(
            similarities_by_class[class_id],
            dtype=np.float32,
        )
        confidences = np.asarray(
            confidences_by_class[class_id],
            dtype=np.float32,
        )
        all_similarities.append(similarities)
        all_confidences.append(confidences)
        counts, _ = np.histogram(similarities, bins=bins)
        axis.bar(
            centers,
            counts,
            width=widths,
            color="#E45756",
            edgecolor="white",
            linewidth=0.8,
        )
        axis.axvline(
            args.similarity_reference,
            color="#1F1F1F",
            linestyle="--",
            linewidth=2,
            label=f"reference={args.similarity_reference:.2f}",
        )
        ratio = float(
            (similarities > args.similarity_reference).mean()
        )
        axis.set_title(
            (
                f"{CLASS_NAMES[class_id]}  (n={len(similarities)})\n"
                f"mean={similarities.mean():.3f}, "
                f">0.7={ratio * 100:.1f}%"
            ),
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlabel(
            "Maximum similarity to 10 prototypes of predicted class"
        )
        axis.set_ylabel("Clear negative sample count")
        axis.set_xlim(-1.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(loc="upper left", frameon=False, fontsize=9)

        high_confidence = confidences >= 0.2
        summary_rows.append(
            {
                "class": CLASS_NAMES[class_id],
                "clear_negative_count": len(similarities),
                "mean_similarity": float(similarities.mean()),
                "median_similarity": float(np.median(similarities)),
                "p10_similarity": float(np.quantile(similarities, 0.1)),
                "p90_similarity": float(np.quantile(similarities, 0.9)),
                "ratio_similarity_gt_0_7": ratio,
                "high_confidence_count": int(high_confidence.sum()),
                "high_confidence_mean_similarity": (
                    float(similarities[high_confidence].mean())
                    if np.any(high_confidence)
                    else np.nan
                ),
                "high_confidence_ratio_similarity_gt_0_7": (
                    float(
                        (
                            similarities[high_confidence]
                            > args.similarity_reference
                        ).mean()
                    )
                    if np.any(high_confidence)
                    else np.nan
                ),
            }
        )

    axes[-1].axis("off")
    figure.suptitle(
        (
            "Clear-negative similarity distribution, iter_5000\n"
            "unmatched, nearest GT center distance > 2 m, "
            "7 active predicted classes, no confidence threshold"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    image_path = os.path.join(
        args.out_dir,
        "clear_negative_similarity_hist_7cls_iter5000.png",
    )
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    overall_similarity = np.concatenate(all_similarities)
    overall_confidence = np.concatenate(all_confidences)
    overall_high_confidence = overall_confidence >= 0.2
    summary_rows.append(
        {
            "class": "overall",
            "clear_negative_count": len(overall_similarity),
            "mean_similarity": float(overall_similarity.mean()),
            "median_similarity": float(np.median(overall_similarity)),
            "p10_similarity": float(
                np.quantile(overall_similarity, 0.1)
            ),
            "p90_similarity": float(
                np.quantile(overall_similarity, 0.9)
            ),
            "ratio_similarity_gt_0_7": float(
                (
                    overall_similarity
                    > args.similarity_reference
                ).mean()
            ),
            "high_confidence_count": int(
                overall_high_confidence.sum()
            ),
            "high_confidence_mean_similarity": float(
                overall_similarity[overall_high_confidence].mean()
            ),
            "high_confidence_ratio_similarity_gt_0_7": float(
                (
                    overall_similarity[overall_high_confidence]
                    > args.similarity_reference
                ).mean()
            ),
        }
    )
    summary_path = os.path.join(
        args.out_dir,
        "clear_negative_similarity_summary_7cls_iter5000.csv",
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
            f"{row['class']}: n={row['clear_negative_count']}, "
            f"mean={row['mean_similarity']:.4f}, "
            f"median={row['median_similarity']:.4f}, "
            f">0.7={row['ratio_similarity_gt_0_7'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
