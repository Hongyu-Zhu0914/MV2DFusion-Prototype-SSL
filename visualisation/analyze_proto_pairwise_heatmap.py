#!/usr/bin/env python3
"""Draw pairwise cosine-similarity heatmap for class prototypes.

Default behavior matches the previous 7-class/70-prototype heatmap:
car, truck, bus, motorcycle, bicycle, pedestrian, traffic_cone.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
DEFAULT_KEEP_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description='Draw pairwise prototype cosine-similarity heatmap.')
    parser.add_argument('--proto-ckpt', required=True, help='Checkpoint containing fusion_bbox_head.prototypes or prototypes')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--tag', default='proto_pairwise', help='Filename tag for outputs')
    parser.add_argument('--keep-classes', nargs='+', default=DEFAULT_KEEP_CLASSES,
                        help='Classes to include. Default gives 7 classes / 70 prototypes')
    parser.add_argument('--vmin', type=float, default=-1.0, help='Heatmap minimum value')
    parser.add_argument('--vmax', type=float, default=1.0, help='Heatmap maximum value')
    parser.add_argument('--figsize', nargs=2, type=float, default=[18.0, 16.0], help='Figure size: width height')
    parser.add_argument('--dpi', type=int, default=180, help='Figure DPI')
    return parser.parse_args()


def l2_normalize(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def load_prototypes(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        for key, value in ckpt['state_dict'].items():
            if key.endswith('prototypes') and torch.is_tensor(value) and value.ndim == 3:
                return value.detach().float().cpu().numpy(), key
    if isinstance(ckpt, dict) and 'prototypes' in ckpt:
        value = ckpt['prototypes']
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        return value, 'prototypes'
    raise RuntimeError(f'Cannot find prototypes in {path}')


def compute_pairwise(prototypes, keep_classes):
    keep_indices = [NAME_TO_IDX[name] for name in keep_classes]
    prototypes = l2_normalize(prototypes.astype(np.float32), axis=-1)
    selected = prototypes[keep_indices].reshape(-1, prototypes.shape[-1])
    sim = selected @ selected.T
    return sim, selected, keep_indices


def make_labels(keep_classes, num_prototype):
    labels = []
    for cls_name in keep_classes:
        prefix = cls_name[:4]
        for proto_idx in range(num_prototype):
            labels.append(f'{prefix}{proto_idx}')
    return labels


def save_heatmap(sim, keep_classes, num_prototype, out_png, title, vmin, vmax, figsize, dpi):
    labels = make_labels(keep_classes, num_prototype)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(sim, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    total = len(labels)
    for boundary in range(0, total + 1, num_prototype):
        ax.axhline(boundary - 0.5, color='black', linewidth=1.0)
        ax.axvline(boundary - 0.5, color='black', linewidth=1.0)

    for class_idx, cls_name in enumerate(keep_classes):
        mid = class_idx * num_prototype + (num_prototype - 1) / 2
        ax.text(mid, -4.2, cls_name, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(-4.2, mid, cls_name, ha='right', va='center', fontsize=9, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='cosine similarity')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def save_stats(sim, keep_classes, num_prototype, proto_path, proto_key, proto_shape, selected_shape, out_path):
    lines = [
        f'ckpt: {proto_path}',
        f'proto_key: {proto_key}',
        f'shape_all: {proto_shape}',
        f'shape_selected: {selected_shape}',
        f'classes: {keep_classes}',
        '',
    ]
    for class_idx, cls_name in enumerate(keep_classes):
        block = sim[class_idx*num_prototype:(class_idx+1)*num_prototype,
                    class_idx*num_prototype:(class_idx+1)*num_prototype]
        off_diag = block[~np.eye(num_prototype, dtype=bool)]
        lines.append(f'intra,{cls_name},mean={off_diag.mean():.6f},min={off_diag.min():.6f},max={off_diag.max():.6f}')
    for i, cls_i in enumerate(keep_classes):
        for j, cls_j in enumerate(keep_classes):
            if j <= i:
                continue
            block = sim[i*num_prototype:(i+1)*num_prototype,
                        j*num_prototype:(j+1)*num_prototype]
            lines.append(f'inter,{cls_i}-{cls_j},mean={block.mean():.6f},max={block.max():.6f},min={block.min():.6f}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prototypes, proto_key = load_prototypes(args.proto_ckpt)
    num_prototype = prototypes.shape[1]
    sim, selected, _ = compute_pairwise(prototypes, args.keep_classes)

    out_png = os.path.join(args.out_dir, f'{args.tag}.png')
    out_npy = os.path.join(args.out_dir, f'{args.tag}.npy')
    out_stats = os.path.join(args.out_dir, f'{args.tag}_stats.txt')
    np.save(out_npy, sim)

    title = (
        f'Pairwise cosine similarity of {len(args.keep_classes) * num_prototype} prototypes '
        f'({len(args.keep_classes)} classes x {num_prototype})\n{os.path.basename(args.proto_ckpt)}'
    )
    save_heatmap(sim, args.keep_classes, num_prototype, out_png, title,
                 args.vmin, args.vmax, tuple(args.figsize), args.dpi)
    save_stats(sim, args.keep_classes, num_prototype, args.proto_ckpt, proto_key,
               tuple(prototypes.shape), tuple(selected.shape), out_stats)

    mask = ~np.eye(sim.shape[0], dtype=bool)
    print('Saved outputs:')
    print(' ', out_png)
    print(' ', out_npy)
    print(' ', out_stats)
    print('overall offdiag mean/std/min/max:',
          float(sim[mask].mean()), float(sim[mask].std()), float(sim[mask].min()), float(sim[mask].max()))


if __name__ == '__main__':
    main()
