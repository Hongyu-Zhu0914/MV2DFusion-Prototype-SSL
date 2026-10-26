"""Evaluate a prototype checkpoint on the approved custom nuScenes val subset.

Unlike the stock nuScenes evaluator, ground truth is restricted to exactly the
tokens represented by the validation PKL. This keeps mAP/NDS comparable across
checkpoints without treating the held-out half of official val as misses.
"""

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
import os
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                          load_gt, load_prediction)
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval


class SubsetDetectionEval(DetectionEval):
    """Official detection metric calculation on an explicit set of val tokens."""

    def __init__(self, nusc, config, result_path, output_dir, sample_tokens):
        self.nusc = nusc
        self.result_path = str(result_path)
        self.eval_set = 'val'
        self.output_dir = str(output_dir)
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        self.verbose = False
        self.cfg = config
        os.makedirs(self.plot_dir, exist_ok=True)

        expected = set(sample_tokens)
        if not expected or len(expected) != len(sample_tokens):
            raise ValueError('Validation token list is empty or has duplicates')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path, config.max_boxes_per_sample, DetectionBox,
            verbose=False)
        predicted = set(self.pred_boxes.sample_tokens)
        if predicted != expected:
            raise ValueError('Prediction token mismatch: missing={}, extra={}'.format(
                len(expected - predicted), len(predicted - expected)))

        full_gt = load_gt(nusc, 'val', DetectionBox, verbose=False)
        available = set(full_gt.sample_tokens)
        if not expected <= available:
            raise ValueError('Subset contains {} tokens outside official val'.format(
                len(expected - available)))
        self.gt_boxes = EvalBoxes()
        for token in sample_tokens:
            self.gt_boxes.add_boxes(token, full_gt[token])
        if set(self.gt_boxes.sample_tokens) != predicted:
            raise AssertionError('Restricted GT and predictions differ')

        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes,
                                            config.class_range, verbose=False)
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes,
                                          config.class_range, verbose=False)
        self.sample_tokens = self.gt_boxes.sample_tokens
        if set(self.sample_tokens) != expected or set(self.pred_boxes.sample_tokens) != expected:
            raise AssertionError('Filtering changed validation token coverage')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--baseline', action='store_true',
                       help='Use full R50 detector weights, only for smoke testing')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-samples', type=int, default=3014)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only for an isolated small-sample smoke test')
    args = parser.parse_args()
    if args.limit is not None and not 0 < args.limit <= args.expected_samples:
        raise ValueError('Invalid smoke-test limit')
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('Refusing to overwrite an existing validation directory')
    output_dir.mkdir(parents=True, exist_ok=True)

    import projects.mmdet3d_plugin
    import projects.fsdv2
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = build_dataset(cfg.data.val)
    if len(dataset) != args.expected_samples:
        raise AssertionError('Unexpected custom-val length: {}'.format(len(dataset)))
    if args.limit is not None:
        dataset.data_infos = dataset.data_infos[:args.limit]
    tokens = [row['token'] for row in dataset.data_infos]
    if len(tokens) != len(set(tokens)) or len(tokens) != len(dataset):
        raise AssertionError('Dataset tokens are missing or duplicated')
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              num_gpus=1, dist=False, shuffle=False,
                              nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    model = build_model(cfg.model)
    if args.baseline:
        model.init_weights()
    else:
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    outputs = single_gpu_test(model, loader)
    if len(outputs) != len(dataset):
        raise AssertionError('Inference did not return one result per sample')
    result_files, tmp_dir = dataset.format_results(
        outputs, jsonfile_prefix=str(output_dir / 'predictions'))
    if tmp_dir is not None:
        raise AssertionError('Unexpected temporary prediction directory')
    if isinstance(result_files, dict):
        result_path = result_files.get('pts_bbox', next(iter(result_files.values())))
    else:
        result_path = result_files

    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root,
                    verbose=False)
    evaluator = SubsetDetectionEval(
        nusc, dataset.eval_detection_configs, result_path,
        output_dir / 'nuscenes_metrics', tokens)
    metrics = evaluator.main(render_curves=False)
    summary = {
        'frames': len(dataset),
        'token_sha256': hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest(),
        'prediction_json': str(result_path),
        'checkpoint': str(args.checkpoint) if args.checkpoint else 'weights/r50.pth',
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    }
    (output_dir / 'subset_validation_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print('SUBSET_VALIDATION_PASS', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
