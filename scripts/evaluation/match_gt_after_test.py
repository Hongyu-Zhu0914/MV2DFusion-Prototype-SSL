
# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.utils import import_modules_from_strings


sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, (str(REPO_ROOT / 'projects')))


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用按类别贪心匹配为 NMS 特征匹配 GT 标签')
    parser.add_argument('--config', type=str, required=True,
                        help='测试时使用的配置文件路径')
    parser.add_argument('--pkl', type=str, required=True,
                        help='PKL 文件路径（包含 infos）')
    parser.add_argument(
        '--feature-dir', type=str,
        default=(str(REPO_ROOT / 'work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/features/legacy_nms_features_all')),
        help='特征文件目录（基础目录）')
    parser.add_argument('--dist-th', type=float, default=0.5,
                        help='中心点距离阈值（米）')
    parser.add_argument(
        '--proto-file', type=str,
        default='work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/prototypes/prototypes_iter_40000.pth',
        help='原型文件或完整 checkpoint 路径')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='置信度阈值，用于筛选负样本')
    parser.add_argument('--sim-thr', type=float, default=0.5,
                        help='余弦相似度阈值，用于筛选负样本')
    return parser.parse_args()


def load_pkl_infos(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        infos = data.get('infos', data)
    else:
        infos = data
    print(f"✅ 从 PKL 文件中加载了 {len(infos)} 个样本")
    return infos


def sigmoid(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def load_prototypes(path):
    proto_data = torch.load(path, map_location='cpu')
    prototypes = None
    if isinstance(proto_data, dict) and 'state_dict' in proto_data:
        for key, value in proto_data['state_dict'].items():
            if key.endswith('.prototypes') and torch.is_tensor(value):
                prototypes = value
                break
    elif isinstance(proto_data, dict) and 'prototypes' in proto_data:
        prototypes = proto_data['prototypes']
    if prototypes is None:
        raise RuntimeError(f'未在原型文件中找到 prototypes: {path}')
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().float().cpu().numpy()
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if prototypes.ndim != 3:
        raise RuntimeError(f'原型形状应为 [C,K,D]，实际为 {prototypes.shape}')
    return prototypes


def greedy_match_by_class(boxes, scores, gt_boxes, gt_labels, dist_th,
                          all_dists):
    """按类别和置信度进行一对一贪心中心距离匹配。"""
    num_pred = len(boxes)
    probs = sigmoid(scores)
    confidences = probs.max(axis=1)
    pred_classes = probs.argmax(axis=1)

    nearest_gt_dist = np.min(all_dists, axis=1)
    nearest_gt_class = gt_labels[np.argmin(all_dists, axis=1)]
    matched_labels = -np.ones(num_pred, dtype=np.int64)
    matched_gt_set = set()
    gt_matched_info = {}

    for class_id in range(len(CLASS_NAMES)):
        pred_indices = np.where(pred_classes == class_id)[0]
        gt_indices = np.where(gt_labels == class_id)[0]
        if len(pred_indices) == 0 or len(gt_indices) == 0:
            continue
        pred_indices = sorted(
            pred_indices.tolist(), key=lambda idx: confidences[idx],
            reverse=True)
        matched_gt_in_class = set()
        for pred_idx in pred_indices:
            best_gt_idx = -1
            best_dist = float('inf')
            for gt_idx in gt_indices:
                gt_idx = int(gt_idx)
                if gt_idx in matched_gt_in_class:
                    continue
                dist = float(all_dists[pred_idx, gt_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0 and best_dist < dist_th:
                matched_labels[pred_idx] = class_id
                matched_gt_in_class.add(best_gt_idx)
                matched_gt_set.add(best_gt_idx)
                gt_matched_info[best_gt_idx] = (pred_idx, best_dist)

    return (matched_labels, matched_gt_set, gt_matched_info,
            nearest_gt_dist, nearest_gt_class)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print('✅ 配置文件加载成功')
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
        print('✅ 已导入自定义模块')

    infos = load_pkl_infos(args.pkl)
    base_dir = args.feature_dir
    gt_dir = os.path.join(base_dir, f'th_{args.dist_th}')
    os.makedirs(gt_dir, exist_ok=True)
    print(f'📂 基础特征目录: {base_dir}')
    print(f'📂 GT 输出目录: {gt_dir}')
    print(f'🎯 距离阈值: {args.dist_th}m')

    prototypes = load_prototypes(args.proto_file)
    num_classes, num_prototypes, dim = prototypes.shape
    proto_norm = l2_normalize(prototypes.reshape(-1, dim))
    print(f'✅ 加载原型成功，形状: {prototypes.shape}')

    feature_files = sorted(
        [name for name in os.listdir(base_dir)
         if name.startswith('sample_') and name.endswith('_features.npy')],
        key=lambda name: int(name.split('_')[1]))
    if not feature_files:
        raise RuntimeError(f'未找到特征文件: {base_dir}')
    print(f'📁 找到 {len(feature_files)} 个特征文件')

    success_count = 0
    for feat_file in feature_files:
        sample_idx = int(feat_file.split('_')[1])
        if sample_idx >= len(infos):
            print(f'⚠️ 样本索引 {sample_idx} 超出 PKL 范围，跳过')
            continue
        info = infos[sample_idx]
        token = info['token']
        feat_path = os.path.join(base_dir, feat_file)
        boxes_path = os.path.join(
            base_dir, f'sample_{sample_idx}_boxes.npy')
        scores_path = os.path.join(
            base_dir, f'sample_{sample_idx}_scores.npy')
        gt_path = os.path.join(
            gt_dir, f'sample_{sample_idx}_gt_labels.npy')
        token_path = os.path.join(
            gt_dir, f'sample_{sample_idx}_token.npy')
        fn_path = os.path.join(gt_dir, f'sample_{sample_idx}_fn.npy')

        if not all(os.path.exists(path) for path in
                   (feat_path, boxes_path, scores_path)):
            print(f'⚠️ 样本 {sample_idx} 缺少必要文件，跳过')
            continue

        try:
            features = np.load(feat_path).astype(np.float32)
            boxes = np.load(boxes_path).astype(np.float32)
            scores = np.load(scores_path).astype(np.float32)
            if not (len(features) == len(boxes) == len(scores)):
                raise RuntimeError(
                    f'长度不一致: features={len(features)}, '
                    f'boxes={len(boxes)}, scores={len(scores)}')

            gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
            gt_names = np.asarray(info['gt_names'])
            valid = np.array([name in NAME_TO_IDX for name in gt_names])
            gt_boxes = gt_boxes[valid]
            gt_names = gt_names[valid]
            gt_labels = np.array(
                [NAME_TO_IDX[name] for name in gt_names], dtype=np.int64)

            print(f'\n🔍 处理样本 {sample_idx}, token: {token[:8]}...')
            print(f'   预测框: {len(boxes)} 个')
            print(f'   GT 框: {len(gt_boxes)} 个')

            if len(gt_boxes) == 0:
                matched_labels = -np.ones(len(boxes), dtype=np.int64)
                np.save(gt_path, matched_labels)
                np.save(token_path, np.array([token], dtype=object))
                np.save(fn_path, np.array([0], dtype=np.int64))
                success_count += 1
                continue

            all_dists = np.linalg.norm(
                boxes[:, :2, None] - gt_boxes[:, :2].T, axis=1)
            (matched_labels, matched_gt_set, gt_matched_info,
             nearest_dist, _) = greedy_match_by_class(
                boxes, scores, gt_boxes, gt_labels, args.dist_th, all_dists)

            fn_count = len(gt_boxes) - len(matched_gt_set)
            print(f'   ✅ 贪心匹配完成: {(matched_labels >= 0).sum()}/'
                  f'{len(boxes)} 个查询有 GT')
            print(f'   📊 FN 数量: {fn_count}/{len(gt_boxes)}')
            np.save(gt_path, matched_labels)
            np.save(token_path, np.array([token], dtype=object))
            np.save(fn_path, np.array([fn_count], dtype=np.int64))

            features_norm = l2_normalize(features)
            cosine = (features_norm @ proto_norm.T).reshape(
                len(boxes), num_classes, num_prototypes)
            max_sim_per_class = cosine.max(axis=2)
            probs = sigmoid(scores)
            pred_classes = probs.argmax(axis=1)
            pred_conf = probs.max(axis=1)
            sim_to_pred_class = max_sim_per_class[
                np.arange(len(boxes)), pred_classes]

            high_quality_neg = (
                (matched_labels == -1) &
                (sim_to_pred_class > args.sim_thr) &
                (pred_conf > args.score_thr))
            if np.any(high_quality_neg):
                indices = np.where(high_quality_neg)[0]
                indices = sorted(
                    indices, key=lambda idx: pred_conf[idx], reverse=True)
                print(f'   🔍 高相似度负样本: {len(indices)} 个')
                for pred_idx in indices:
                    nearest_gt_idx = int(np.argmin(all_dists[pred_idx]))
                    gt_cls = int(gt_labels[nearest_gt_idx])
                    matched = gt_matched_info.get(nearest_gt_idx)
                    matched_text = (
                        f'{matched[1]:.2f}m (查询 {matched[0]})'
                        if matched is not None else 'None')
                    print(
                        f'     查询 {pred_idx:4d}: '
                        f'预测类别={CLASS_NAMES[pred_classes[pred_idx]]:20} '
                        f'置信度={pred_conf[pred_idx]:.4f}, '
                        f'余弦相似度={sim_to_pred_class[pred_idx]:.4f}, '
                        f'最近 GT={CLASS_NAMES[gt_cls]} '
                        f'距离={nearest_dist[pred_idx]:.2f}m, '
                        f'该 GT 已匹配查询距离={matched_text}')

            fn_gt_indices = set(range(len(gt_boxes))) - matched_gt_set
            if fn_gt_indices:
                print(f'   🔴 FN GT 列表: {len(fn_gt_indices)} 个')
                for gt_idx in sorted(fn_gt_indices):
                    gt_cls = int(gt_labels[gt_idx])
                    distances = all_dists[:, gt_idx]
                    nearest_idx = int(np.argmin(distances))
                    same_class = pred_classes == gt_cls
                    if np.any(same_class):
                        candidates = np.where(same_class)[0]
                        same_idx = int(candidates[np.argmin(distances[candidates])])
                        same_text = (
                            f'查询 {same_idx}, 距离={distances[same_idx]:.2f}m')
                    else:
                        same_text = '无同类查询'
                    print(
                        f'     GT {gt_idx}: 类别={CLASS_NAMES[gt_cls]}, '
                        f'全局最近查询={nearest_idx}, '
                        f'距离={distances[nearest_idx]:.2f}m, '
                        f'同类最近={same_text}')

            success_count += 1
        except Exception as exc:
            print(f'   ❌ 处理失败: {exc}')
            import traceback
            traceback.print_exc()

    print(f'\n🎉 处理完成！成功处理 {success_count}/{len(feature_files)} 个样本')


if __name__ == '__main__':
    main()
