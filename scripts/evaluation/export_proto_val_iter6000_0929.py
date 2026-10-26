"""Test the exact trained prototype model and export aligned query features."""

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
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet3d.core.bbox.structures import xywhr2xyxyr  # noqa: F401; plugin import order
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes import NuScenes

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from scripts.evaluation.validate_prototype_subset_0924 import SubsetDetectionEval


CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)


def match_at_two_meters(boxes, scores, info):
    """One-to-one same-class greedy match for legacy clear-negative plots."""
    names = np.asarray(info['gt_names'])
    valid = np.isin(names, CLASSES)
    gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)[valid]
    gt_labels = np.asarray([CLASSES.index(str(x)) for x in names[valid]], dtype=np.int64)
    matched = np.full(len(boxes), -1, dtype=np.int64)
    if not len(boxes) or not len(gt_boxes):
        return matched
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    distances = np.linalg.norm(boxes[:, None, :2] - gt_boxes[None, :, :2], axis=2)
    for class_id in range(len(CLASSES)):
        prediction_ids = np.where(predicted == class_id)[0]
        gt_ids = set(np.where(gt_labels == class_id)[0].tolist())
        for prediction_id in sorted(prediction_ids, key=lambda j: -confidence[j]):
            eligible = [(distances[prediction_id, j], j) for j in gt_ids
                        if distances[prediction_id, j] < 2.0]
            if eligible:
                _, gt_id = min(eligible)
                matched[prediction_id] = class_id
                gt_ids.remove(gt_id)
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, help='Small isolated smoke test')
    parser.add_argument('--skip-metrics', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Refusing to overwrite output directory: {}'.format(root))
    root.mkdir(parents=True, exist_ok=True)
    features_dir = root / 'features'
    matched_dir = features_dir / 'th_2.0'
    matched_dir.mkdir(parents=True)

    import projects.mmdet3d_plugin
    import projects.fsdv2
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = build_dataset(cfg.data.val)
    if len(dataset) != 3014:
        raise AssertionError('Expected the approved 3014-frame val split')
    if args.limit is not None:
        if not 0 < args.limit <= len(dataset):
            raise ValueError('Invalid smoke-test limit')
        dataset.data_infos = dataset.data_infos[:args.limit]
    tokens = [info['token'] for info in dataset.data_infos]
    if len(tokens) != len(set(tokens)):
        raise AssertionError('Duplicate sample tokens')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              num_gpus=1, dist=False, shuffle=False,
                              nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = dataset.CLASSES
    model.cuda().eval()
    head = model.fusion_bbox_head
    if head.post_bev_nms_ops != [0]:
        raise AssertionError('Expected the trained NMS policy')
    last_features = []
    head_outputs = []

    def capture_classifier_input(_, inputs):
        last_features.append(inputs[0].detach().clone())

    def capture_head_output(_, __, output):
        head_outputs.append(output)

    pre_handle = head.cls_branches[-1][-1].register_forward_pre_hook(
        capture_classifier_input)
    out_handle = head.register_forward_hook(capture_head_output)
    wrapped = MMDataParallel(model, device_ids=[0])
    outputs = []
    manifest = []
    total_queries = 0
    total_valid = 0
    try:
        for index, batch in enumerate(loader):
            last_features.clear()
            head_outputs.clear()
            with torch.no_grad():
                result = wrapped(return_loss=False, rescale=True, **batch)
                if len(result) != 1 or not last_features or len(head_outputs) != 1:
                    raise AssertionError('Expected one result/head and classifier features; got '
                                         'result={}, features={}, head={}'.format(
                                             len(result), len(last_features), len(head_outputs)))
                head_result = head_outputs[0]
                score_tensor = head_result['all_cls_scores'][-1, 0]
                box_tensor = head_result['all_bbox_preds'][-1, 0]
                hidden = last_features[-1]
                batch_size, query_count, channels = hidden.shape
                projected = head.proto_proj(hidden.reshape(batch_size * query_count, channels))
                projected = projected.reshape(batch_size, query_count, -1)
                projected = F.normalize(head.feat_norm(projected), p=2, dim=-1)[0]
                if score_tensor.shape[0] != box_tensor.shape[0] or score_tensor.shape[0] != projected.shape[0]:
                    raise AssertionError('Query count differs across score, box and feature')
                total_queries += score_tensor.shape[0]
                # The trained head writes exactly -40 to scores rejected by
                # its valid-query mask and BEV NMS. Keep all surviving normal
                # and temporal queries, with no additional score threshold.
                keep = ~(score_tensor <= -39.5).all(dim=-1)
                raw_class_scores = head.cls_branches[-1][-1](last_features[-1])[0]
                if not torch.allclose(raw_class_scores[keep], score_tensor[keep],
                                      rtol=1e-3, atol=1e-3):
                    raise AssertionError('Classifier features do not align with surviving scores')
                query_indices = torch.where(keep)[0]
                valid_scores = score_tensor[keep].float().cpu().numpy()
                valid_features = projected[keep].float().cpu().numpy()
                valid_boxes = denormalize_bbox(box_tensor[keep], None).float().cpu().numpy()
                if not all(len(x) == len(query_indices) for x in
                           (valid_scores, valid_features, valid_boxes)):
                    raise AssertionError('Exported arrays are misaligned')
                if not all(np.isfinite(x).all() for x in
                           (valid_scores, valid_features, valid_boxes)):
                    raise AssertionError('Nonfinite exported prediction')
                if valid_scores.shape[1] != 10 or valid_features.shape[1] != 256:
                    raise AssertionError('Unexpected class or feature dimensions')
                labels = valid_scores.argmax(axis=1).astype(np.int64)
                token = tokens[index]
                prefix = features_dir / 'sample_{}'.format(index)
                np.save(str(prefix) + '_features.npy', valid_features)
                np.save(str(prefix) + '_scores.npy', valid_scores)
                np.save(str(prefix) + '_boxes.npy', valid_boxes)
                np.save(str(prefix) + '_labels.npy', labels)
                np.save(str(prefix) + '_query_indices.npy', query_indices.cpu().numpy())
                np.save(str(prefix) + '_token.npy', np.asarray([token]))
                matched = match_at_two_meters(
                    valid_boxes, valid_scores, dataset.data_infos[index])
                np.save(matched_dir / 'sample_{}_gt_labels.npy'.format(index), matched)
                outputs.extend(result)
                total_valid += len(valid_scores)
                manifest.append(dict(index=index, token=token,
                                     all_queries=int(score_tensor.shape[0]),
                                     valid_queries=len(valid_scores),
                                     normal_queries=int((query_indices < head.num_query).sum()),
                                     temporal_queries=int((query_indices >= head.num_query).sum())))
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print('EXPORT_PROGRESS {}/{} valid_queries={}'.format(
                    index + 1, len(dataset), total_valid), flush=True)
    finally:
        pre_handle.remove()
        out_handle.remove()

    if len(outputs) != len(dataset) or len(manifest) != len(dataset):
        raise AssertionError('Frame coverage mismatch')
    result_pkl = root / 'iter_6000_val_predictions.pkl'
    with result_pkl.open('wb') as file:
        pickle.dump(outputs, file, protocol=pickle.HIGHEST_PROTOCOL)
    with (root / 'feature_manifest.json').open('w') as file:
        json.dump(manifest, file, indent=2)

    summary = dict(frames=len(dataset), checkpoint=str(Path(args.checkpoint).resolve()),
                   prediction_pkl=str(result_pkl), total_queries=total_queries,
                   valid_queries=total_valid,
                   normal_queries=sum(row['normal_queries'] for row in manifest),
                   temporal_queries=sum(row['temporal_queries'] for row in manifest),
                   token_sha256=hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest())
    if not args.skip_metrics:
        result_files, tmp_dir = dataset.format_results(
            outputs, jsonfile_prefix=str(root / 'nuscenes_predictions'))
        if tmp_dir is not None:
            raise AssertionError('Unexpected temporary prediction directory')
        result_path = result_files['pts_bbox'] if isinstance(result_files, dict) else result_files
        nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False)
        evaluator = SubsetDetectionEval(
            nusc, dataset.eval_detection_configs, result_path,
            root / 'nuscenes_metrics', tokens)
        metrics = evaluator.main(render_curves=False)
        summary.update(prediction_json=str(result_path),
                       mAP=float(metrics['mean_ap']), NDS=float(metrics['nd_score']))
    with (root / 'export_summary.json').open('w') as file:
        json.dump(summary, file, indent=2)
    print('EXPORT_PASS {}'.format(json.dumps(summary)), flush=True)


if __name__ == '__main__':
    main()
