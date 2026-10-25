#!/usr/bin/env python3
"""Analyze same-class prototype usage for GT-matched positive predictions.

This script answers a very specific question:
for every GT-matched positive prediction, which same-class prototype is the nearest
one in cosine-similarity space, and how many positives use each prototype?

Outputs include:
  1) per-positive similarity CSV to 10 same-class prototypes;
  2) per-class prototype usage table CSV/TXT;
  3) usage count + percentage visualization;
  4) max-similarity histogram;
  5) mean similarity heatmap to each prototype.
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description='Analyze prototype usage for GT-matched positives.')
    parser.add_argument('--feature-dir', required=True, help='Directory containing sample_*_features/boxes/scores.npy')
    parser.add_argument('--proto-ckpt', required=True, help='Checkpoint containing fusion_bbox_head.prototypes or prototypes')
    parser.add_argument('--pkl', required=True, help='nuScenes info pkl containing gt_boxes and gt_names')
    parser.add_argument('--out-dir', default=None, help='Output directory. Default: <feature-dir>/proto_usage_analysis')
    parser.add_argument('--dist-th', type=float, default=2.0, help='Center-distance threshold for greedy GT matching')
    parser.add_argument('--keep-classes', nargs='+', default=DEFAULT_KEEP_CLASSES,
                        help='Classes to analyze. Default ignores construction_vehicle/trailer/barrier')
    parser.add_argument('--score-mode', choices=['max_sigmoid', 'pred_label_sigmoid'], default='max_sigmoid',
                        help='Confidence used only for greedy ordering. max_sigmoid matches previous analysis.')
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def l2_normalize(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get('infos', data)
    return data


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


def greedy_match_with_gt_idx(boxes, scores, gt_boxes, gt_labels, dist_th, score_mode):
    n_pred = len(boxes)
    matched_cls = -np.ones(n_pred, dtype=np.int64)
    matched_gt_idx = -np.ones(n_pred, dtype=np.int64)
    if n_pred == 0 or len(gt_boxes) == 0:
        return matched_cls, matched_gt_idx

    probs = sigmoid(scores)
    pred_cls = probs.argmax(axis=1)
    if score_mode == 'pred_label_sigmoid':
        conf = probs[np.arange(len(probs)), pred_cls]
    else:
        conf = probs.max(axis=1)
    dists = np.linalg.norm(boxes[:, :2, None] - gt_boxes[:, :2].T, axis=1)

    for cls_id in range(len(CLASS_NAMES)):
        pred_indices = np.where(pred_cls == cls_id)[0]
        gt_indices = np.where(gt_labels == cls_id)[0]
        if len(pred_indices) == 0 or len(gt_indices) == 0:
            continue
        pred_indices = sorted(pred_indices.tolist(), key=lambda idx: conf[idx], reverse=True)
        used_gt = set()
        for pred_idx in pred_indices:
            best_gt = -1
            best_dist = float('inf')
            for gt_idx in gt_indices:
                gt_idx = int(gt_idx)
                if gt_idx in used_gt:
                    continue
                dist = float(dists[pred_idx, gt_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_gt = gt_idx
            if best_gt >= 0 and best_dist < dist_th:
                matched_cls[pred_idx] = cls_id
                matched_gt_idx[pred_idx] = best_gt
                used_gt.add(best_gt)
    return matched_cls, matched_gt_idx


def collect_positive_records(args, prototypes, keep_indices):
    infos = load_infos(args.pkl)
    records = []
    sims_by_cls = defaultdict(list)
    max_by_cls = defaultdict(list)
    mean_by_cls = defaultdict(list)
    best_proto_by_cls = defaultdict(list)
    conf_by_cls = defaultdict(list)

    proto_mean = l2_normalize(prototypes.mean(axis=1), axis=-1)
    feature_files = sorted(
        [p for p in os.listdir(args.feature_dir) if p.startswith('sample_') and p.endswith('_features.npy')],
        key=lambda name: int(name.split('_')[1])
    )

    for feat_name in feature_files:
        sample_idx = int(feat_name.split('_')[1])
        if sample_idx >= len(infos):
            continue
        feat_path = os.path.join(args.feature_dir, feat_name)
        box_path = os.path.join(args.feature_dir, f'sample_{sample_idx}_boxes.npy')
        score_path = os.path.join(args.feature_dir, f'sample_{sample_idx}_scores.npy')
        if not (os.path.exists(box_path) and os.path.exists(score_path)):
            continue

        feats = np.load(feat_path).astype(np.float32)
        boxes = np.load(box_path).astype(np.float32)
        scores = np.load(score_path).astype(np.float32)
        if len(feats) != len(boxes) or len(feats) != len(scores):
            print(f'[WARN] skip sample {sample_idx}: length mismatch feats={len(feats)} boxes={len(boxes)} scores={len(scores)}')
            continue

        info = infos[sample_idx]
        gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
        gt_names = np.asarray(info['gt_names'])
        valid = np.array([name in NAME_TO_IDX for name in gt_names])
        gt_boxes = gt_boxes[valid]
        gt_labels = np.array([NAME_TO_IDX[name] for name in gt_names[valid]], dtype=np.int64)

        matched_cls, matched_gt_idx = greedy_match_with_gt_idx(
            boxes, scores, gt_boxes, gt_labels, args.dist_th, args.score_mode)
        probs = sigmoid(scores)
        pred_cls = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)
        feats = l2_normalize(feats, axis=-1)

        for pred_idx in np.where(matched_gt_idx >= 0)[0]:
            cls_id = int(matched_cls[pred_idx])
            if cls_id not in keep_indices:
                continue
            sim10 = feats[pred_idx] @ prototypes[cls_id].T
            best_idx = int(np.argmax(sim10))
            max_sim = float(sim10[best_idx])
            mean_sim = float(feats[pred_idx] @ proto_mean[cls_id])
            cls_name = CLASS_NAMES[cls_id]
            row = {
                'sample_idx': sample_idx,
                'pred_idx': int(pred_idx),
                'class': cls_name,
                'cls_id': cls_id,
                'gt_idx': int(matched_gt_idx[pred_idx]),
                'pred_cls': int(pred_cls[pred_idx]),
                'confidence': float(pred_conf[pred_idx]),
                'best_proto': best_idx,
                'max_sim': max_sim,
                'proto_mean_sim': mean_sim,
            }
            for proto_idx, sim in enumerate(sim10):
                row[f'sim_proto_{proto_idx}'] = float(sim)
            records.append(row)
            sims_by_cls[cls_id].append(sim10)
            max_by_cls[cls_id].append(max_sim)
            mean_by_cls[cls_id].append(mean_sim)
            best_proto_by_cls[cls_id].append(best_idx)
            conf_by_cls[cls_id].append(float(pred_conf[pred_idx]))

    return records, sims_by_cls, max_by_cls, mean_by_cls, best_proto_by_cls, conf_by_cls


def save_records_csv(records, out_path):
    columns = ['sample_idx', 'pred_idx', 'class', 'cls_id', 'gt_idx', 'pred_cls', 'confidence',
               'best_proto', 'max_sim', 'proto_mean_sim'] + [f'sim_proto_{i}' for i in range(10)]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def save_usage_tables(args, keep_classes, keep_indices, best_proto_by_cls, max_by_cls, mean_by_cls, out_dir):
    table_csv = os.path.join(out_dir, 'positive_nearest_proto_usage_full_table.csv')
    table_txt = os.path.join(out_dir, 'positive_nearest_proto_usage_full_table.txt')
    summary_txt = os.path.join(out_dir, 'positive_similarity_summary.txt')

    lines = [
        f'feature_dir: {args.feature_dir}',
        f'proto_ckpt: {args.proto_ckpt}',
        f'pkl: {args.pkl}',
        f'dist_th: {args.dist_th}',
        f'score_mode: {args.score_mode}',
        '',
    ]
    summary_lines = lines + [
        'class,n,max_sim_mean,max_sim_p10,max_sim_p50,max_sim_p90,proto_mean_sim_mean,proto_mean_sim_p50,used_proto_count,usage_counts_proto0-9'
    ]

    with open(table_csv, 'w') as f:
        f.write('class,total,used_proto_count,dominant_proto,dominant_count,dominant_ratio,' + ','.join([f'proto_{i}' for i in range(10)]) + '\n')
        for cls_name, cls_id in zip(keep_classes, keep_indices):
            best_arr = np.asarray(best_proto_by_cls[cls_id], dtype=np.int64)
            usage = np.bincount(best_arr, minlength=10) if len(best_arr) else np.zeros(10, dtype=np.int64)
            total = int(usage.sum())
            used = int((usage > 0).sum())
            dominant_proto = int(np.argmax(usage)) if total else -1
            dominant_count = int(usage[dominant_proto]) if total else 0
            dominant_ratio = dominant_count / total if total else 0.0
            f.write(f'{cls_name},{total},{used},{dominant_proto},{dominant_count},{dominant_ratio:.6f},' + ','.join(map(str, usage.tolist())) + '\n')

            lines.append(f'{cls_name}: total={total}, used={used}/10, dominant=proto_{dominant_proto} ({dominant_count}/{total}, {dominant_ratio:.2%})')
            for proto_idx, count in enumerate(usage):
                pct = int(count) / total if total else 0.0
                lines.append(f'  proto_{proto_idx}: {int(count):5d}  ({pct:6.2%})')
            lines.append('')

            max_arr = np.asarray(max_by_cls[cls_id], dtype=np.float32)
            mean_arr = np.asarray(mean_by_cls[cls_id], dtype=np.float32)
            if len(max_arr):
                summary_lines.append(
                    f'{cls_name},{len(max_arr)},{max_arr.mean():.6f},{np.percentile(max_arr,10):.6f},'
                    f'{np.percentile(max_arr,50):.6f},{np.percentile(max_arr,90):.6f},'
                    f'{mean_arr.mean():.6f},{np.percentile(mean_arr,50):.6f},{used},{usage.tolist()}'
                )
            else:
                summary_lines.append(f'{cls_name},0,nan,nan,nan,nan,nan,nan,0,{usage.tolist()}')

    with open(table_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(summary_txt, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    return table_csv, table_txt, summary_txt


def plot_usage_table(table_csv, out_path):
    classes, counts, totals = [], [], []
    with open(table_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row['class'])
            arr = [int(row[f'proto_{i}']) for i in range(10)]
            counts.append(arr)
            totals.append(int(row['total']))
    counts = np.asarray(counts, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64)
    perc = counts / (totals[:, None] + 1e-9) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), dpi=180, gridspec_kw={'height_ratios': [1.1, 1.4]})
    ax = axes[0]
    x = np.arange(10)
    bar_w = 0.11
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for cls_idx, cls_name in enumerate(classes):
        ax.bar(x + (cls_idx - (len(classes) - 1) / 2) * bar_w, counts[cls_idx],
               width=bar_w, label=f'{cls_name} (n={totals[cls_idx]})', color=colors[cls_idx], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'proto_{i}' for i in range(10)])
    ax.set_ylabel('Positive sample count')
    ax.set_title('Nearest-prototype usage count for GT-matched positive samples', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncol=4, fontsize=8)

    ax = axes[1]
    im = ax.imshow(perc, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels([f'{cls} (n={n})' for cls, n in zip(classes, totals)])
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f'proto_{i}' for i in range(10)], rotation=30, ha='right')
    ax.set_title('Usage percentage within each class', fontsize=12, fontweight='bold')
    for row_idx in range(counts.shape[0]):
        for proto_idx in range(counts.shape[1]):
            count = counts[row_idx, proto_idx]
            pct = perc[row_idx, proto_idx]
            text = f'{count}\n{pct:.1f}%' if count > 0 else '0'
            ax.text(proto_idx, row_idx, text, ha='center', va='center', fontsize=8,
                    color='black' if pct < 65 else 'white')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='usage percentage (%)')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_max_similarity_hist(keep_classes, keep_indices, max_by_cls, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=160)
    axes = axes.ravel()
    bins = np.linspace(-0.2, 1.0, 49)
    for ax, cls_name, cls_id in zip(axes, keep_classes, keep_indices):
        arr = np.asarray(max_by_cls[cls_id], dtype=np.float32)
        if len(arr):
            ax.hist(arr, bins=bins, color='#4C78A8', alpha=0.85, edgecolor='white')
            ax.axvline(arr.mean(), color='red', linestyle='--', linewidth=1.5, label=f'mean={arr.mean():.3f}')
            ax.axvline(np.median(arr), color='orange', linestyle='-', linewidth=1.3, label=f'median={np.median(arr):.3f}')
        ax.set_title(f'{cls_name} max similarity to 10 prototypes\nn={len(arr)}')
        ax.set_xlabel('max cosine similarity')
        ax.set_ylabel('count')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[len(keep_classes):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_mean_similarity_heatmap(keep_classes, keep_indices, sims_by_cls, out_path):
    mean_sim_mat = np.full((len(keep_indices), 10), np.nan, dtype=np.float32)
    for row_idx, cls_id in enumerate(keep_indices):
        if len(sims_by_cls[cls_id]):
            mean_sim_mat[row_idx] = np.asarray(sims_by_cls[cls_id]).mean(axis=0)
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    im = ax.imshow(mean_sim_mat, cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto')
    ax.set_yticks(np.arange(len(keep_classes)))
    ax.set_yticklabels(keep_classes)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel('prototype index')
    ax.set_title('Mean cosine similarity: positive samples to each same-class prototype')
    for i in range(mean_sim_mat.shape[0]):
        for j in range(mean_sim_mat.shape[1]):
            val = mean_sim_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val < 0.55 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.feature_dir, 'proto_usage_analysis')
    os.makedirs(out_dir, exist_ok=True)
    keep_classes = args.keep_classes
    keep_indices = [NAME_TO_IDX[name] for name in keep_classes]

    prototypes, proto_key = load_prototypes(args.proto_ckpt)
    prototypes = l2_normalize(prototypes.astype(np.float32), axis=-1)
    print(f'Loaded prototypes: key={proto_key}, shape={prototypes.shape}')

    records, sims_by_cls, max_by_cls, mean_by_cls, best_proto_by_cls, _ = collect_positive_records(
        args, prototypes, keep_indices)
    print(f'Collected positive samples: {len(records)}')

    per_sample_csv = os.path.join(out_dir, 'positive_samples_similarity_to_10_prototypes.csv')
    save_records_csv(records, per_sample_csv)

    table_csv, table_txt, summary_txt = save_usage_tables(
        args, keep_classes, keep_indices, best_proto_by_cls, max_by_cls, mean_by_cls, out_dir)

    usage_png = os.path.join(out_dir, 'positive_nearest_proto_usage_full_table.png')
    hist_png = os.path.join(out_dir, 'positive_max_similarity_hist_7cls.png')
    heatmap_png = os.path.join(out_dir, 'positive_mean_similarity_to_each_proto_heatmap.png')
    plot_usage_table(table_csv, usage_png)
    plot_max_similarity_hist(keep_classes, keep_indices, max_by_cls, hist_png)
    plot_mean_similarity_heatmap(keep_classes, keep_indices, sims_by_cls, heatmap_png)

    print('Saved outputs:')
    for path in [per_sample_csv, table_csv, table_txt, summary_txt, usage_png, hist_png, heatmap_png]:
        print(' ', path)


if __name__ == '__main__':
    main()
