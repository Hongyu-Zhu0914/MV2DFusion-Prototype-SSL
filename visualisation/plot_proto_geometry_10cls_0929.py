"""Prototype PCA, class similarity, usage and distance figures for 10 classes."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)


def read_prototypes(path):
    raw = torch.load(path, map_location='cpu')
    if 'prototypes' in raw:
        array = raw['prototypes'].detach().float().numpy()
    else:
        keys = [k for k in raw['state_dict'] if k.endswith('.prototypes')]
        if len(keys) != 1:
            raise AssertionError('Expected one prototype tensor')
        array = raw['state_dict'][keys[0]].detach().float().numpy()
    if array.shape != (10, 10, 256) or not np.isfinite(array).all():
        raise AssertionError('Expected finite [10,10,256] prototypes')
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', required=True)
    parser.add_argument('--last', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = read_prototypes(args.first)
    last = read_prototypes(args.last)

    stacked = np.concatenate((first.reshape(100, 256), last.reshape(100, 256)))
    pca = PCA(n_components=2, random_state=0)
    coordinates = pca.fit_transform(stacked)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), dpi=180, sharex=True, sharey=True)
    colors = plt.cm.tab10(np.arange(10))
    for axis, step, values in zip(axes, (1000, 6000),
                                  (coordinates[:100], coordinates[100:])):
        for cls in range(10):
            points = values[cls * 10:(cls + 1) * 10]
            axis.scatter(points[:, 0], points[:, 1], s=40, color=colors[cls],
                         label=CLASSES[cls], alpha=0.85)
            axis.text(*points.mean(axis=0), str(cls), fontsize=10, weight='bold')
        axis.set_title('Iteration {}'.format(step))
        axis.set_xlabel('PC1 ({:.1%} variance)'.format(pca.explained_variance_ratio_[0]))
        axis.set_ylabel('PC2 ({:.1%} variance)'.format(pca.explained_variance_ratio_[1]))
        axis.grid(alpha=0.2)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.suptitle('Class prototypes in a shared PCA space')
    fig.tight_layout()
    fig.savefig(out / 'prototype_pca_iter1000_vs_iter6000_10cls.png', bbox_inches='tight')
    plt.close(fig)

    matrix = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        for j in range(10):
            sims = last[i] @ last[j].T
            if i == j:
                sims = sims[np.triu_indices(10, k=1)]
            matrix[i, j] = sims.mean()
    fig, axis = plt.subplots(figsize=(12, 10), dpi=180)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
    axis.set_xticks(range(10), CLASSES, rotation=50, ha='right')
    axis.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            axis.text(j, i, '{:.2f}'.format(matrix[i, j]), ha='center', va='center',
                      fontsize=8, color='white' if abs(matrix[i, j]) > 0.6 else 'black')
    axis.set_title('Mean cosine between class prototypes at iteration 6000\n'
                   '(diagonal excludes self-comparisons)')
    fig.colorbar(image, ax=axis, label='Cosine similarity')
    fig.tight_layout()
    fig.savefig(out / 'prototype_class_similarity_heatmap_10cls.png', bbox_inches='tight')
    plt.close(fig)
    np.save(out / 'prototype_class_similarity_matrix_10cls.npy', matrix)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint['state_dict']
    keys = [k for k in state if k.endswith('.proto_update_count')]
    if len(keys) != 1:
        raise AssertionError('Expected one prototype update counter')
    counts = state[keys[0]].cpu().numpy()
    if counts.shape != (10, 10):
        raise AssertionError('Expected ten classes and ten prototypes')
    fig, axis = plt.subplots(figsize=(12, 9), dpi=180)
    image = axis.imshow(counts, cmap='viridis', aspect='auto')
    axis.set_xticks(range(10), ['P{}'.format(x) for x in range(10)])
    axis.set_yticks(range(10), CLASSES)
    axis.set_title('Prototype update counts at iteration 6000')
    fig.colorbar(image, ax=axis, label='Updates')
    fig.tight_layout()
    fig.savefig(out / 'prototype_update_count_heatmap_10cls.png', bbox_inches='tight')
    plt.close(fig)

    rows = []
    for cls in range(10):
        within = np.linalg.norm(last[cls][:, None] - last[cls][None, :], axis=-1)
        other = np.concatenate([last[j] for j in range(10) if j != cls])
        nearest = np.linalg.norm(last[cls][:, None] - other[None, :], axis=-1).min(axis=1)
        rows.append(dict(class_name=CLASSES[cls], update_count=int(counts[cls].sum()),
                         within_class_mean_distance=float(within[np.triu_indices(10, 1)].mean()),
                         nearest_other_class_mean_distance=float(nearest.mean()),
                         within_class_mean_cosine=float(matrix[cls, cls])))
    with (out / 'prototype_distance_attributes_10cls.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    x = np.arange(10)
    fig, axis = plt.subplots(figsize=(15, 6), dpi=180)
    axis.bar(x - 0.2, [row['within_class_mean_distance'] for row in rows],
             width=0.4, label='Within-class prototype distance')
    axis.bar(x + 0.2, [row['nearest_other_class_mean_distance'] for row in rows],
             width=0.4, label='Nearest different-class prototype distance')
    axis.set_xticks(x, CLASSES, rotation=45, ha='right')
    axis.set_ylabel('Euclidean distance between L2-normalized prototypes')
    axis.set_title('Prototype distance attributes at iteration 6000')
    axis.legend()
    axis.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'prototype_distance_attributes_10cls.png', bbox_inches='tight')
    plt.close(fig)
    print('PROTO_GEOMETRY_PASS', out, flush=True)


if __name__ == '__main__':
    main()
