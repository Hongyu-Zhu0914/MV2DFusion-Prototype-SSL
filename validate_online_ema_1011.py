"""Evaluate one SSL branch on the exact labeled custom-val token subset."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import types

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet.models import build_detector
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from validate_prototype_subset_0924 import SubsetDetectionEval
from train_online_ema_1011 import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/nusc/ssl_online_r50_ema_1011.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--baseline', action='store_true')
    parser.add_argument('--branch', choices=['student', 'teacher'], default='student')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config); setup(cfg)
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Refusing to overwrite evaluation results')
    directory.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(cfg.data.val)
    if len(dataset) != 3014:
        raise AssertionError('Expected 3014 labeled custom-val frames')
    if args.limit:
        if not 0 < args.limit <= 3014:
            raise ValueError('Invalid smoke limit')
        dataset.data_infos = dataset.data_infos[:args.limit]
    tokens = [row['token'] for row in dataset.data_infos]
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                             num_gpus=1, dist=False, shuffle=False,
                             shuffler_sampler=cfg.data.shuffler_sampler,
                             nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    architecture = copy.deepcopy(Config.fromfile(cfg.model.detector_config).model)
    model = build_detector(architecture)
    if args.baseline:
        load_checkpoint(model, cfg.model.initial_checkpoint, map_location='cpu', strict=True)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        prefix = args.branch+'.'
        full_state = checkpoint['state_dict']
        state = OrderedDict((k[len(prefix):], v) for k,v in full_state.items() if k.startswith(prefix))
        # spconv uses module version metadata to distinguish sparse kernel layouts.
        # Preserve and rebase it when extracting one branch from the wrapper.
        if not hasattr(full_state, '_metadata'):
            raise RuntimeError('SSL checkpoint lacks module-version metadata')
        state._metadata = OrderedDict(
            ('' if k == args.branch else k[len(prefix):], v)
            for k,v in full_state._metadata.items()
            if k == args.branch or k.startswith(prefix))
        if not state:
            raise ValueError('Missing requested SSL branch')
        # Use the same MMCV loader as the baseline. Torch's generic loader can
        # trigger spconv legacy-layout conversion after branch metadata stripping.
        load_state_dict(model, state, strict=True)
        del checkpoint, state, full_state
    # Match the current online framework's per-frame temporal reset.
    original = model.simple_test
    def single_frame(self, *inputs, **kwargs):
        self.fusion_bbox_head.reset_memory(); self.prev_scene_token = None; self.test_clip_id = 1
        output = original(*inputs, **kwargs)
        self.fusion_bbox_head.reset_memory()
        return output
    model.simple_test = types.MethodType(single_frame, model)
    model.CLASSES = dataset.CLASSES
    outputs = single_gpu_test(MMDataParallel(model.cuda().eval(), device_ids=[0]), loader)
    if len(outputs) != len(tokens):
        raise AssertionError('Output/token count mismatch')
    mmcv.dump(outputs, str(directory/'predictions.pkl'))
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=str(directory/'predictions'))
    path = result_files['pts_bbox'] if isinstance(result_files, dict) else result_files
    nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False)
    metrics = SubsetDetectionEval(nusc, dataset.eval_detection_configs, path,
                                  directory/'nuscenes_metrics', tokens).main(render_curves=False)
    summary = dict(frames=len(tokens), branch=args.branch, checkpoint=args.checkpoint,
                   baseline=args.baseline, smoke_only=bool(args.limit), temporal_mode='reset_each_frame',
                   token_sha256=hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest(),
                   mAP=metrics['mean_ap'], NDS=metrics['nd_score'])
    if not all(torch.isfinite(torch.tensor(summary[k])) for k in ['mAP','NDS']):
        raise AssertionError('Non-finite metrics')
    (directory/'subset_validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('SSL_SUBSET_EVAL_PASS', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
