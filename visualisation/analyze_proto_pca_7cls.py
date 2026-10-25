#!/usr/bin/env python3
"""Visualize GT-matched positive samples and prototypes with per-class PCA.

The script generates two PCA views by default:
  1) with_proto_fit: PCA is fitted by positives + 10 prototypes + prototype mean;
  2) sample_fit_proto_projected: PCA is fitted only by positives, then prototypes are projected.

Each mode outputs seven per-class figures and one combined 7-class figure.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from shapely.geometry import Polygon

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
DEFAULT_KEEP_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
VALID_MODES = ['with_proto_fit', 'sample_fit_proto_projected']


def parse_args():
    parser = argparse.ArgumentParser(description='Draw 7-class PCA figures for positives and prototypes.')
    parser.add_argument('--feature-dir', required=True, help='Directory containing sample_*_features/boxes/scores.npy')
    parser.add_argument('--proto-ckpt', required=True, help='Checkpoint containing fusion_bbox_head.prototypes or prototypes')
    parser.add_argument('--pkl', required=True, help='nuScenes info pkl containing gt_boxes and gt_names')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--dist-th', type=float, default=2.0, help='Center-distance threshold for greedy GT matching')
    parser.add_argument('--modes', nargs='+', choices=VALID_MODES + ['both'], default=['both'],
                        help='PCA modes to generate. Default: both')
    parser.add_argument('--keep-classes', nargs='+', default=DEFAULT_KEEP_CLASSES,
                        help='Classes to include. Default ignores construction_vehicle/trailer/barrier')
    parser.add_argument('--score-mode', choices=['max_sigmoid', 'pred_label_sigmoid'], default='max_sigmoid',
                        help='Confidence used only for greedy ordering. max_sigmoid matches previous analysis.')
    parser.add_argument('--overlap-denom', choices=['pred', 'gt'], default='pred',
                        help='Color value denominator: intersection/pred_area or intersection/gt_area. Default matches previous pred-area version.')
    parser.add_argument('--sample-size', type=float, default=26.0, help='Scatter point size for positive samples')
    parser.add_argument('--dpi', type=int, default=180, help='Figure DPI')
    parser.add_argument('--axis-limit', type=float, default=None,
                        help='Fix both PCA axes to [-axis_limit, axis_limit].')
    parser.add_argument('--coordinate-scale', type=float, default=1.0,
                        help='Divide all projected PCA coordinates by this shared scale before plotting.')
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


def get_bev_corners(box):
    cx, cy = float(box[0]), float(box[1])
    w, l = float(box[3]), float(box[4])
    yaw = float(box[6])
    corners = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


def overlap_ratio(pred_box, gt_box, denom):
    pred_poly = Polygon(get_bev_corners(pred_box))
    gt_poly = Polygon(get_bev_corners(gt_box))
    if not pred_poly.is_valid or not gt_poly.is_valid:
        return 0.0
    inter = pred_poly.intersection(gt_poly).area
    base = pred_poly.area if denom == 'pred' else gt_poly.area
    if base <= 0:
        return 0.0
    return float(np.clip(inter / base, 0.0, 1.0))


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


def collect_positive_features(args, keep_indices):
    infos = load_infos(args.pkl)
    features_by_cls = defaultdict(list)
    overlap_by_cls = defaultdict(list)
    conf_by_cls = defaultdict(list)

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
        pred_conf = sigmoid(scores).max(axis=1)

        for pred_idx in np.where(matched_gt_idx >= 0)[0]:
            cls_id = int(matched_cls[pred_idx])
            if cls_id not in keep_indices:
                continue
            gt_idx = int(matched_gt_idx[pred_idx])
            features_by_cls[cls_id].append(feats[pred_idx])
            overlap_by_cls[cls_id].append(overlap_ratio(boxes[pred_idx], gt_boxes[gt_idx], args.overlap_denom))
            conf_by_cls[cls_id].append(float(pred_conf[pred_idx]))

    return features_by_cls, overlap_by_cls, conf_by_cls


def draw_one_class(cls_name, feats, overlaps, protos, center, mode, out_path,
                   sample_size, dpi, axis_limit, coordinate_scale):
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=dpi)
    if len(feats) < 2:
        ax.text(0.5, 0.5, f'Insufficient positives: {len(feats)}', ha='center', va='center')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return None

    if mode == 'with_proto_fit':
        combined = np.concatenate([feats, protos, center], axis=0)
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        feat_2d = combined_2d[:len(feats)]
        proto_2d = combined_2d[len(feats):len(feats) + len(protos)]
        center_2d = combined_2d[-1]
        title_mode = 'PCA fitted by positives + prototypes + proto mean'
    else:
        pca = PCA(n_components=2)
        feat_2d = pca.fit_transform(feats)
        proto_2d = pca.transform(protos)
        center_2d = pca.transform(center)[0]
        title_mode = 'PCA fitted by positives only; prototypes projected'

    if coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be positive')
    feat_2d = feat_2d / coordinate_scale
    proto_2d = proto_2d / coordinate_scale
    center_2d = center_2d / coordinate_scale

    sc = ax.scatter(
        feat_2d[:, 0], feat_2d[:, 1], c=overlaps, cmap=plt.cm.YlOrRd,
        norm=plt.Normalize(0, 1), s=sample_size, alpha=0.80, edgecolors='none',
        label='GT-matched samples', zorder=3)
    zero_overlap = overlaps == 0
    if zero_overlap.any():
        ax.scatter(
            feat_2d[zero_overlap, 0], feat_2d[zero_overlap, 1],
            c='lightblue', s=sample_size, alpha=0.80, edgecolors='none',
            zorder=4)
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], marker='*', c='red', s=260,
               edgecolors='black', linewidths=0.9, label='10 prototypes', zorder=8)
    for proto_idx, xy in enumerate(proto_2d):
        ax.text(xy[0], xy[1], str(proto_idx), fontsize=8, color='black', ha='center', va='center', zorder=9)
    ax.scatter(center_2d[0], center_2d[1], marker='s', c='yellow', s=220,
               edgecolors='black', linewidths=1.2, label='prototype mean', zorder=10)
    ax.set_title(f'{cls_name}: {title_mode}\nn={len(feats)}', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    if axis_limit is not None:
        if axis_limit <= 0:
            raise ValueError('axis_limit must be positive')
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ticks = np.linspace(-axis_limit, axis_limit, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.22)
    ax.legend(loc='best', fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Overlap ratio')
    cbar.set_ticks(np.linspace(0, 1, 6))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return {
        'features_2d': feat_2d,
        'prototypes_2d': proto_2d,
        'center_2d': center_2d,
        'coordinate_scale': coordinate_scale,
        'axis_limit': axis_limit,
        'pca_explained_variance_ratio': pca.explained_variance_ratio_,
        'proto_center_dist_over_sample_std': float(np.linalg.norm(center_2d - feat_2d.mean(0)) / (feat_2d.std(0).mean() + 1e-9)),
    }


def save_combined(image_paths, out_path, title):
    imgs = [Image.open(path).convert('RGB') for path in image_paths]
    if not imgs:
        return
    thumb_w, thumb_h = 750, 625
    margin, title_h = 24, 54
    cols, rows = 4, 2
    canvas = Image.new('RGB', (cols * thumb_w + (cols + 1) * margin,
                               rows * thumb_h + (rows + 1) * margin + title_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), title, fill='black')
    for idx, img in enumerate(imgs):
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (idx % cols) * (thumb_w + margin) + (thumb_w - img.width) // 2
        y = title_h + margin + (idx // cols) * (thumb_h + margin) + (thumb_h - img.height) // 2
        canvas.paste(img, (x, y))
    canvas.save(out_path)


def main():
    args = parse_args()
    modes = VALID_MODES if 'both' in args.modes else args.modes
    os.makedirs(args.out_dir, exist_ok=True)
    keep_classes = args.keep_classes
    keep_indices = [NAME_TO_IDX[name] for name in keep_classes]

    prototypes, proto_key = load_prototypes(args.proto_ckpt)
    prototypes = l2_normalize(prototypes.astype(np.float32), axis=-1)
    proto_centers = l2_normalize(prototypes.mean(axis=1), axis=-1)
    print(f'Loaded prototypes: key={proto_key}, shape={prototypes.shape}')

    features_by_cls, overlap_by_cls, conf_by_cls = collect_positive_features(args, keep_indices)
    print('Loaded positives:')
    for cls_id in keep_indices:
        print(f'  {CLASS_NAMES[cls_id]:16s}: {len(features_by_cls[cls_id])}')

    stats_lines = ['mode,class,n_pos,overlap_mean,overlap_median,pca_var1,pca_var2,proto_center_dist_over_sample_std,output']
    for mode in modes:
        mode_dir = os.path.join(args.out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        image_paths = []
        for cls_name, cls_id in zip(keep_classes, keep_indices):
            feats = np.asarray(features_by_cls[cls_id], dtype=np.float32)
            overlaps = np.asarray(overlap_by_cls[cls_id], dtype=np.float32)
            protos = prototypes[cls_id].astype(np.float32)
            center = proto_centers[cls_id].reshape(1, -1).astype(np.float32)
            out_path = os.path.join(mode_dir, f'{cls_name}_{mode}.png')
            result = draw_one_class(
                cls_name, feats, overlaps, protos, center, mode, out_path,
                args.sample_size, args.dpi, args.axis_limit,
                args.coordinate_scale)
            image_paths.append(out_path)
            if result is None:
                stats_lines.append(f'{mode},{cls_name},{len(feats)},nan,nan,nan,nan,nan,{out_path}')
                continue
            np.save(os.path.join(mode_dir, f'{cls_name}_{mode}_points.npy'), {
                'features_2d': result['features_2d'],
                'prototypes_2d': result['prototypes_2d'],
                'center_2d': result['center_2d'],
                'coordinate_scale': result['coordinate_scale'],
                'axis_limit': result['axis_limit'],
                'overlap_ratio': overlaps,
                'conf': np.asarray(conf_by_cls[cls_id], dtype=np.float32),
                'pca_explained_variance_ratio': result['pca_explained_variance_ratio'],
            })
            stats_lines.append(
                f'{mode},{cls_name},{len(feats)},{overlaps.mean():.6f},{np.median(overlaps):.6f},'
                f'{result["pca_explained_variance_ratio"][0]:.6f},{result["pca_explained_variance_ratio"][1]:.6f},'
                f'{result["proto_center_dist_over_sample_std"]:.6f},{out_path}'
            )
        combined_title = 'PCA fitted with positives + prototypes + prototype mean' if mode == 'with_proto_fit' else 'PCA fitted with positives only; prototypes projected'
        save_combined(image_paths, os.path.join(args.out_dir, f'{mode}_combined.png'), combined_title)

    stats_path = os.path.join(args.out_dir, 'pca_compare_stats.csv')
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats_lines) + '\n')
    print('Saved root:', args.out_dir)
    print('Saved stats:', stats_path)
    for mode in modes:
        print('Saved combined:', os.path.join(args.out_dir, f'{mode}_combined.png'))


if __name__ == '__main__':
    main()

